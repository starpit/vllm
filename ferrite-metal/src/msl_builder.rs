/// MSL source code builder.
///
/// Rust equivalent of MFA's CodeWriter. Used at COMPILE TIME by the proc macro
/// to generate MSL kernel source strings. No runtime dependencies.
///
/// Like PtxBuilder on the CUDA side, this is a pure Rust library that emits
/// strings. The proc macro calls it, gets an MSL string back, and embeds it
/// as a const in the generated code.
use std::collections::HashMap;
use std::fmt::Write;

pub struct MslBuilder {
    /// The MSL source being built.
    source: String,
    /// Template variables for {{KEY}} substitution.
    vars: HashMap<String, String>,
    /// Current indentation level.
    indent: u32,
}

impl MslBuilder {
    pub fn new() -> Self {
        Self {
            source: String::with_capacity(16384),
            vars: HashMap::new(),
            indent: 0,
        }
    }

    /// Set a template variable for {{KEY}} substitution.
    pub fn set(&mut self, key: &str, value: impl ToString) {
        self.vars.insert(key.to_string(), value.to_string());
    }

    /// Append a raw line (no template substitution).
    pub fn raw(&mut self, line: &str) {
        self.write_indent();
        self.source.push_str(line);
        self.source.push('\n');
    }

    /// Append a line with {{KEY}} template substitution.
    pub fn line(&mut self, template: &str) {
        self.write_indent();
        self.source.push_str(&self.substitute(template));
        self.source.push('\n');
    }

    /// Append a block of text with template substitution.
    /// Each line gets indented.
    pub fn block(&mut self, template: &str) {
        for line in template.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                self.source.push('\n');
            } else {
                self.write_indent();
                self.source.push_str(&self.substitute(trimmed));
                self.source.push('\n');
            }
        }
    }

    /// Append raw text without indentation or newlines.
    pub fn append(&mut self, text: &str) {
        self.source.push_str(text);
    }

    /// Increase indent level.
    pub fn indent(&mut self) {
        self.indent += 1;
    }

    /// Decrease indent level.
    pub fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    /// Emit an opening brace and indent.
    pub fn open_brace(&mut self) {
        self.raw("{");
        self.indent();
    }

    /// Dedent and emit a closing brace.
    pub fn close_brace(&mut self) {
        self.dedent();
        self.raw("}");
    }

    /// Emit a blank line.
    pub fn blank(&mut self) {
        self.source.push('\n');
    }

    /// Emit a comment.
    pub fn comment(&mut self, text: &str) {
        self.write_indent();
        write!(self.source, "// {}\n", text).unwrap();
    }

    /// Get the final MSL source string.
    pub fn finish(self) -> String {
        self.source
    }

    /// Perform {{KEY}} substitution on a template string.
    fn substitute(&self, template: &str) -> String {
        let mut result = String::with_capacity(template.len());
        let mut rest = template;
        while let Some(start) = rest.find("{{") {
            result.push_str(&rest[..start]);
            let after = &rest[start + 2..];
            if let Some(end) = after.find("}}") {
                let key = &after[..end];
                if let Some(val) = self.vars.get(key) {
                    result.push_str(val);
                } else {
                    panic!("MslBuilder: undefined template variable '{{{{{}}}}}'", key);
                }
                rest = &after[end + 2..];
            } else {
                result.push_str("{{");
                rest = after;
            }
        }
        result.push_str(rest);
        result
    }

    fn write_indent(&mut self) {
        for _ in 0..self.indent {
            self.source.push_str("    ");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_substitution() {
        let mut b = MslBuilder::new();
        b.set("TYPE", "half");
        b.set("SIZE", "32");
        b.line("threadgroup {{TYPE}} smem[{{SIZE}}];");
        assert_eq!(b.finish().trim(), "threadgroup half smem[32];");
    }

    #[test]
    fn test_indentation() {
        let mut b = MslBuilder::new();
        b.raw("kernel void test() {");
        b.indent();
        b.raw("int x = 0;");
        b.dedent();
        b.raw("}");
        let s = b.finish();
        assert!(s.contains("    int x = 0;"));
    }
}
