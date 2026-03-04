// vLLM WebGPU — Chat application glue
//
// Loads the WASM module, manages UI state, drives inference, and updates
// the gears panel with live stats.

let wasm = null;
let generating = false;

// DOM elements
const modelSelect = document.getElementById('model-select');
const loadBtn = document.getElementById('load-btn');
const loadStatus = document.getElementById('load-status');
const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');

// Gears panel elements
const statTps = document.getElementById('stat-tps');
const statPosition = document.getElementById('stat-position');
const statLastToken = document.getElementById('stat-last-token');
const statProb = document.getElementById('stat-prob');
const statKv = document.getElementById('stat-kv');
const cacheBar = document.getElementById('cache-bar');
const statMemory = document.getElementById('stat-memory');
const layerTimes = document.getElementById('layer-times');

// -------------------------------------------------------------------------
// WASM initialization
// -------------------------------------------------------------------------

async function initWasm() {
    try {
        loadStatus.textContent = 'Loading WASM...';
        // Import the WASM module (built by wasm-pack with --target web)
        const mod = await import('./pkg/vllm_examples.js');
        await mod.default(); // Initialize WASM
        wasm = mod;

        loadStatus.textContent = 'Initializing WebGPU...';
        await wasm.init_device();
        loadStatus.textContent = 'Ready — select a model';
        loadBtn.disabled = false;
    } catch (e) {
        loadStatus.textContent = `Error: ${e}`;
        console.error('WASM init failed:', e);
    }
}

// -------------------------------------------------------------------------
// Model loading
// -------------------------------------------------------------------------

loadBtn.addEventListener('click', async () => {
    if (!wasm) return;
    const modelId = modelSelect.value;
    loadStatus.textContent = `Fetching config for ${modelId}...`;
    loadBtn.disabled = true;

    try {
        // Fetch config.json from HuggingFace
        const configUrl = `https://huggingface.co/${modelId}/resolve/main/config.json`;
        const resp = await fetch(configUrl);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const configJson = await resp.text();

        loadStatus.textContent = 'Parsing config...';
        await wasm.load_model(configJson);

        loadStatus.textContent = `${modelId} loaded (config only — weights via safetensors coming soon)`;
        sendBtn.disabled = false;
    } catch (e) {
        loadStatus.textContent = `Load failed: ${e}`;
        console.error('Model load failed:', e);
    } finally {
        loadBtn.disabled = false;
    }
});

// -------------------------------------------------------------------------
// Chat UI
// -------------------------------------------------------------------------

function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const roleLabel = document.createElement('div');
    roleLabel.className = 'role';
    roleLabel.textContent = role;
    div.appendChild(roleLabel);
    const content = document.createElement('div');
    content.textContent = text;
    div.appendChild(content);
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return content;
}

function updateStats() {
    if (!wasm) return;
    try {
        const stats = JSON.parse(wasm.get_stats());
        statTps.textContent = stats.tokens_per_sec.toFixed(1);
        statPosition.textContent = stats.seq_position;
        statLastToken.textContent = stats.last_token || '—';
        statProb.textContent = stats.last_token_prob > 0
            ? (stats.last_token_prob * 100).toFixed(1) + '%'
            : '—';
        statKv.textContent = `${stats.kv_cache_used} / ${stats.kv_cache_total}`;
        const pct = stats.kv_cache_total > 0
            ? (stats.kv_cache_used / stats.kv_cache_total * 100)
            : 0;
        cacheBar.style.width = `${pct}%`;

        if (stats.gpu_memory_bytes > 0) {
            statMemory.textContent = `${(stats.gpu_memory_bytes / 1024 / 1024).toFixed(1)} MB`;
        }

        // Layer times
        if (stats.layer_times_ms && stats.layer_times_ms.length > 0) {
            layerTimes.innerHTML = stats.layer_times_ms
                .map((t, i) => `<div class="layer-time-row"><span>L${i}</span><span>${t.toFixed(1)} ms</span></div>`)
                .join('');
        }
    } catch (e) {
        // Stats not available yet
    }
}

sendBtn.addEventListener('click', startGeneration);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) startGeneration();
    }
});

stopBtn.addEventListener('click', () => {
    generating = false;
});

async function startGeneration() {
    if (!wasm || generating) return;

    const prompt = userInput.value.trim();
    if (!prompt) return;

    userInput.value = '';
    addMessage('user', prompt);
    generating = true;
    sendBtn.style.display = 'none';
    stopBtn.style.display = '';

    // Reset engine state
    wasm.reset();

    // Simple tokenization placeholder: encode as char codes
    // In production this would use a real tokenizer (e.g., via tokenizers WASM)
    const promptTokens = new Uint32Array([1]); // BOS token
    wasm.set_prompt(promptTokens);

    const contentEl = addMessage('assistant', '');
    let generatedText = '';
    const startTime = performance.now();
    let tokenCount = 0;

    try {
        while (generating && tokenCount < 256) {
            const tokenId = await wasm.generate_next();
            tokenCount++;

            // Simple decode placeholder (would use real tokenizer)
            const char = String.fromCharCode(tokenId % 128 || 32);
            generatedText += char;
            contentEl.textContent = generatedText;

            // Update stats
            const elapsed = (performance.now() - startTime) / 1000;
            statTps.textContent = elapsed > 0 ? (tokenCount / elapsed).toFixed(1) : '—';
            updateStats();

            // Check for EOS (token 2 is common EOS)
            if (tokenId === 2) break;

            // Yield to browser event loop
            await new Promise(r => setTimeout(r, 0));
        }
    } catch (e) {
        contentEl.textContent += `\n[Error: ${e}]`;
        console.error('Generation error:', e);
    }

    generating = false;
    sendBtn.style.display = '';
    stopBtn.style.display = 'none';
    updateStats();
}

// -------------------------------------------------------------------------
// Stats polling
// -------------------------------------------------------------------------

setInterval(updateStats, 500);

// -------------------------------------------------------------------------
// Boot
// -------------------------------------------------------------------------

initWasm();
