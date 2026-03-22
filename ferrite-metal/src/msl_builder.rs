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
    fn test_template_substitution_basic() {
        let mut b = MslBuilder::new();
        b.set("TYPE", "half");
        b.set("SIZE", "32");
        b.line("threadgroup {{TYPE}} smem[{{SIZE}}];");
        assert_eq!(b.finish().trim(), "threadgroup half smem[32];");
    }

    #[test]
    fn test_template_multiple_same_key() {
        let mut b = MslBuilder::new();
        b.set("T", "float");
        b.line("{{T}} a; {{T}} b;");
        assert_eq!(b.finish().trim(), "float a; float b;");
    }

    #[test]
    fn test_template_no_substitution() {
        let mut b = MslBuilder::new();
        b.line("int x = 42;");
        assert_eq!(b.finish().trim(), "int x = 42;");
    }

    #[test]
    #[should_panic(expected = "undefined template variable")]
    fn test_template_undefined_key_panics() {
        let mut b = MslBuilder::new();
        b.line("{{UNDEFINED_KEY}}");
    }

    #[test]
    fn test_template_overwrite() {
        let mut b = MslBuilder::new();
        b.set("X", "1");
        b.line("int a = {{X}};");
        b.set("X", "2");
        b.line("int b = {{X}};");
        let s = b.finish();
        assert!(s.contains("int a = 1;"));
        assert!(s.contains("int b = 2;"));
    }

    #[test]
    fn test_indentation_levels() {
        let mut b = MslBuilder::new();
        b.raw("level0");
        b.indent();
        b.raw("level1");
        b.indent();
        b.raw("level2");
        b.dedent();
        b.raw("level1again");
        b.dedent();
        b.raw("level0again");
        let s = b.finish();
        assert!(s.contains("level0\n"));
        assert!(s.contains("    level1\n"));
        assert!(s.contains("        level2\n"));
        assert!(s.contains("    level1again\n"));
        assert!(s.contains("level0again\n"));
    }

    #[test]
    fn test_dedent_does_not_go_negative() {
        let mut b = MslBuilder::new();
        b.dedent(); // should not panic
        b.dedent(); // should not panic
        b.raw("still at level 0");
        assert!(b.finish().starts_with("still at level 0"));
    }

    #[test]
    fn test_open_close_brace() {
        let mut b = MslBuilder::new();
        b.raw("if (true)");
        b.open_brace();
        b.raw("do_something();");
        b.close_brace();
        let s = b.finish();
        assert!(s.contains("if (true)\n"));
        assert!(s.contains("{\n"));
        assert!(s.contains("    do_something();\n"));
        assert!(s.contains("}\n"));
    }

    #[test]
    fn test_comment() {
        let mut b = MslBuilder::new();
        b.indent();
        b.comment("this is a comment");
        let s = b.finish();
        assert!(s.contains("    // this is a comment\n"));
    }

    #[test]
    fn test_blank_line() {
        let mut b = MslBuilder::new();
        b.raw("line1");
        b.blank();
        b.raw("line2");
        let s = b.finish();
        assert!(s.contains("line1\n\nline2\n"));
    }

    #[test]
    fn test_block_multiline() {
        let mut b = MslBuilder::new();
        b.set("N", "8");
        b.indent();
        b.block("for (int i = 0; i < {{N}}; i++) {\n    x += i;\n}");
        let s = b.finish();
        assert!(s.contains("    for (int i = 0; i < 8; i++) {"));
        assert!(s.contains("    x += i;"));
        assert!(s.contains("    }"));
    }

    #[test]
    fn test_append_no_newline() {
        let mut b = MslBuilder::new();
        b.append("no_newline");
        b.append("_continued");
        let s = b.finish();
        assert_eq!(s, "no_newline_continued");
    }

    #[test]
    fn test_realistic_kernel_structure() {
        let mut b = MslBuilder::new();
        b.set("TYPE", "half");
        b.set("BLOCK_M", "32");
        b.set("BLOCK_N", "32");

        b.raw("#include <metal_stdlib>");
        b.raw("using namespace metal;");
        b.blank();
        b.line("constant uint M_group = {{BLOCK_M}};");
        b.line("constant uint N_group = {{BLOCK_N}};");
        b.blank();
        b.raw("kernel void gemm(");
        b.indent();
        b.line("device {{TYPE}} *A [[buffer(0)]],");
        b.line("device {{TYPE}} *B [[buffer(1)]],");
        b.raw("device float *C [[buffer(2)]]");
        b.dedent();
        b.raw(")");
        b.open_brace();
        b.comment("K-loop");
        b.raw("for (uint k = 0; k < K; k += K_group)");
        b.open_brace();
        b.raw("// body");
        b.close_brace();
        b.close_brace();

        let s = b.finish();
        assert!(s.contains("#include <metal_stdlib>"));
        assert!(s.contains("constant uint M_group = 32;"));
        assert!(s.contains("device half *A [[buffer(0)]],"));
        assert!(s.contains("    // K-loop"));
        assert!(s.contains("    for (uint k = 0; k < K; k += K_group)"));
        assert!(s.contains("        // body"));
    }
}
