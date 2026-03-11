// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! SSH client — tunnel, exec, and file transfer over russh.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use russh::client::{self, AuthResult, Handle};
use russh::keys::{PrivateKey, PrivateKeyWithHashAlg, PublicKey};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

pub fn humanize_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

struct Client;

impl client::Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &PublicKey,
    ) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

/// SSH session wrapper — tunnel, exec, and file transfer.
pub struct SshSession {
    handle: Arc<Mutex<Option<Handle<Client>>>>,
}

impl SshSession {
    /// Connect and authenticate via SSH key.
    pub async fn connect(host: &str, port: u16) -> Result<Self> {
        let key_path = Self::find_ssh_key()?;
        let key_pair = Self::load_private_key(&key_path).await?;

        let username = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "user".to_string());

        let config = Arc::new(client::Config {
            inactivity_timeout: Some(std::time::Duration::from_secs(300)),
            keepalive_interval: Some(std::time::Duration::from_secs(20)),
            keepalive_max: 3,
            ..<_>::default()
        });

        use russh::keys::HashAlg;
        let key_pair = Arc::new(key_pair);

        let mut authenticated_session = None;
        for attempt in 1..=30 {
            // Connect.
            let mut session = match client::connect(config.clone(), (host, port), Client {}).await {
                Ok(s) => s,
                Err(e) => {
                    if attempt >= 30 {
                        return Err(anyhow::anyhow!("failed to connect after 30 retries: {e}"));
                    }
                    if attempt <= 3 || attempt % 5 == 0 {
                        eprintln!("SSH attempt {attempt}/30 failed: {e}. Retrying in 5s...");
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    continue;
                }
            };

            // Authenticate.
            let hash_alg = if key_pair.algorithm().is_rsa() {
                Some(HashAlg::Sha256)
            } else {
                None
            };
            let key_with_alg = PrivateKeyWithHashAlg::new(key_pair.clone(), hash_alg);
            match session
                .authenticate_publickey(&username, key_with_alg)
                .await
            {
                Ok(AuthResult::Success) => {
                    authenticated_session = Some(session);
                    break;
                }
                Ok(AuthResult::Failure { .. }) | Err(_) => {
                    if attempt >= 30 {
                        return Err(anyhow::anyhow!("SSH auth rejected after 30 attempts"));
                    }
                    if attempt <= 3 || attempt % 5 == 0 {
                        eprintln!("SSH auth attempt {attempt}/30 failed. Retrying in 5s...");
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }

        Ok(Self {
            handle: Arc::new(Mutex::new(authenticated_session)),
        })
    }

    /// Upload a directory to the remote host via `tar` pipe.
    /// Excludes `target/` and `.claude/` directories.
    /// Returns (file_count, total_bytes) of the compressed tar.
    pub async fn upload_dir(
        &self,
        local_path: &std::path::Path,
        remote_path: &str,
        progress: Option<&indicatif::ProgressBar>,
    ) -> Result<(usize, usize)> {
        let guard = self.handle.lock().await;
        let session = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("SSH session closed"))?;

        // Open exec channel: mkdir + extract tar on remote side.
        let channel = session.channel_open_session().await?;
        channel
            .exec(
                true,
                format!("mkdir -p {remote_path} && tar xzf - -C {remote_path}"),
            )
            .await?;

        // Create tar locally, tracking file count.
        let local_path = local_path.to_path_buf();
        let pb = progress.cloned();
        let (tar_data, file_count) =
            tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, usize)> {
                let mut count = 0usize;
                let mut encoder =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                {
                    let mut tar = tar::Builder::new(&mut encoder);
                    tar.follow_symlinks(false);

                    let walker = ignore::WalkBuilder::new(&local_path)
                        .hidden(false) // don't skip dotfiles by default
                        .git_ignore(true) // obey .gitignore
                        .git_global(true)
                        .git_exclude(true)
                        .build();

                    for entry in walker.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        let rel = path.strip_prefix(&local_path).unwrap_or(path);
                        if rel.as_os_str().is_empty() {
                            continue;
                        }
                        if path.is_file() {
                            tar.append_path_with_name(path, rel)?;
                            count += 1;
                            if let Some(ref pb) = pb {
                                pb.set_message(format!(
                                    "Packing: {} ({count} files)",
                                    rel.display()
                                ));
                            }
                        } else if path.is_dir() {
                            tar.append_dir(rel, path)?;
                        }
                    }
                    tar.finish()?;
                }
                let data = encoder.finish()?;
                Ok((data, count))
            })
            .await??;

        let total_bytes = tar_data.len();
        if let Some(pb) = progress {
            pb.set_message(format!(
                "Uploading {count} files ({})...",
                humanize_bytes(total_bytes),
                count = file_count,
            ));
        }

        // Write tar data to channel.
        let mut stream = channel.into_stream();
        stream.write_all(&tar_data).await?;
        stream.shutdown().await?;

        Ok((file_count, total_bytes))
    }

    /// Start a TCP port-forwarding tunnel. Returns a handle that accepts connections
    /// in the background. Call `close()` on the returned `SshTunnel` to stop.
    pub fn tunnel(
        self: &Arc<Self>,
        local_port: u16,
        remote_host: String,
        remote_port: u16,
    ) -> SshTunnel {
        SshTunnel {
            session: self.clone(),
            local_port,
            remote_host,
            remote_port,
        }
    }

    /// Close the session.
    pub async fn close(&self) -> Result<()> {
        let mut handle = self.handle.lock().await;
        if let Some(session) = handle.take() {
            session
                .disconnect(russh::Disconnect::ByApplication, "", "en")
                .await
                .context("failed to disconnect")?;
        }
        Ok(())
    }

    fn find_ssh_key() -> Result<PathBuf> {
        let home = std::env::var("HOME").context("HOME not set")?;
        let home_path = PathBuf::from(home);
        for name in ["google_compute_engine", "id_ed25519", "id_rsa", "id_ecdsa"] {
            let path = home_path.join(".ssh").join(name);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(anyhow::anyhow!("no SSH key found in ~/.ssh/"))
    }

    async fn load_private_key(path: &PathBuf) -> Result<PrivateKey> {
        let data = tokio::fs::read_to_string(path)
            .await
            .context("failed to read SSH key")?;
        russh::keys::decode_secret_key(&data, None).context("failed to decode SSH key")
    }
}

/// TCP port-forwarding tunnel over an SSH session.
pub struct SshTunnel {
    session: Arc<SshSession>,
    local_port: u16,
    remote_host: String,
    remote_port: u16,
}

impl SshTunnel {
    /// Start accepting connections on the local port (runs forever).
    pub async fn start(&self) -> Result<()> {
        let listener = TcpListener::bind(("127.0.0.1", self.local_port)).await?;

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let session = self.session.clone();
                    let remote_host = self.remote_host.clone();
                    let remote_port = self.remote_port;
                    tokio::spawn(async move {
                        if let Err(e) =
                            Self::handle_connection(&session, stream, &remote_host, remote_port)
                                .await
                        {
                            eprintln!("tunnel connection error: {e}");
                        }
                    });
                }
                Err(e) => {
                    eprintln!("accept error: {e}");
                }
            }
        }
    }

    async fn handle_connection(
        session: &SshSession,
        mut local_stream: TcpStream,
        remote_host: &str,
        remote_port: u16,
    ) -> Result<()> {
        let channel = {
            let guard = session.handle.lock().await;
            let handle = guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("SSH session closed"))?;
            handle
                .channel_open_direct_tcpip(remote_host, remote_port as u32, "127.0.0.1", 0)
                .await
                .context("failed to open SSH channel")?
        };

        let mut channel_stream = channel.into_stream();
        let _ = tokio::io::copy_bidirectional(&mut local_stream, &mut channel_stream).await;
        Ok(())
    }

    /// Close the underlying SSH session.
    pub async fn close(&self) -> Result<()> {
        self.session.close().await
    }
}
