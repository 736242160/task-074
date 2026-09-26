#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minic.py —— 迷你语言 -> 中间表示(IR) 的单文件编译工具（仅用 Python 标准库）

======================== 语言语法（EBNF） ========================
    program        := statement*
    statement      := decl | assign | ifStmt | whileStmt | block
    decl           := "let" IDENT "=" expr ";"        # 变量声明（先声明后使用）
    assign         := IDENT "=" expr ";"              # 赋值
    ifStmt         := "if" expr block [ "else" block ]
    whileStmt      := "while" expr block
    block          := "{" statement* "}"
    expr           := equality
    equality       := relational ( ("==" | "!=") relational )*
    relational     := additive ( ("<" | ">" | "<=" | ">=") additive )*
    additive       := multiplicative ( ("+" | "-") multiplicative )*
    multiplicative := unary ( ("*" | "/" | "%") unary )*
    unary          := "-" unary | primary
    primary        := INT | IDENT | "(" expr ")"

注释：支持 // 与 # 行注释。变量作用域为全局（块不引入新作用域）。

======================== 中间表示(IR)设计 ========================
采用三地址码，每条指令至多一个运算，分三类：

  1. 运算类：ADD/SUB/MUL/DIV/MOD/NEG/LT/GT/LE/GE/EQ/NE  dst, src1[, src2]
       表达式按优先级拆解为指令序列，中间结果放入临时变量 t0,t1,...；
       比较运算产生 0/1，供条件跳转使用。
  2. 存储类：STORE var, src
       声明与赋值统一为“把值写入变量”的存储指令，与运算解耦。
  3. 跳转类：LABEL L / JMP L / JZ cond, L
       结构化的 if/while 降级为线性跳转；JZ 在条件为 0 时跳转。
       标签编号在生成时新鲜分配，嵌套结构由递归下降自然保证
       “后开先关”的配对关系（内层标签完整落在在层标签之间）。

设计理由：三地址码接近真实汇编/虚拟机中间层，便于后续做常量折叠、
死代码消除等优化以及目标代码生成；“算、存、控”三类指令语义正交，
易于实现、检查与验证。

======================== 用法 ========================
    python3 minic.py 文件路径     # 编译文件
    python3 minic.py -            # 从标准输入读取
    python3 minic.py --demo       # 运行内置示例（正确示例 + 错误示例）
"""
import sys
from dataclasses import dataclass

KEYWORDS = {"let", "if", "else", "while"}

BINOPS = {
    "+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "%": "MOD",
    "<": "LT", ">": "GT", "<=": "LE", ">=": "GE", "==": "EQ", "!=": "NE",
}


class ParseError(Exception):
    """语句级恐慌模式恢复的内部信号，错误信息已记录。"""


@dataclass
class Token:
    kind: str   # 'INT' / 'IDENT' / 关键字 / 运算符字面量 / 'EOF'
    value: str
    line: int


def tokenize(source, errors):
    """把源文本切成 token 序列；词法错误记录到 errors 并跳过该字符。"""
    tokens = []
    i, line, n = 0, 1, len(source)
    two_char = ("<=", ">=", "==", "!=")
    one_char = set("+-*/%<>=;{}()")
    while i < n:
        ch = source[i]
        if ch in " \t\r":
            i += 1
        elif ch == "\n":
            line += 1
            i += 1
        elif ch == "#" or (ch == "/" and i + 1 < n and source[i + 1] == "/"):
            while i < n and source[i] != "\n":
                i += 1
        elif ch.isdigit():
            j = i
            while j < n and source[j].isdigit():
                j += 1
            tokens.append(Token("INT", source[i:j], line))
            i = j
        elif ch.isalpha() or ch == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            tokens.append(Token(word if word in KEYWORDS else "IDENT", word, line))
            i = j
        elif source[i:i + 2] in two_char:
            tokens.append(Token(source[i:i + 2], source[i:i + 2], line))
            i += 2
        elif ch in one_char:
            tokens.append(Token(ch, ch, line))
            i += 1
        elif ch == "!":
            errors.append(f"第 {line} 行: 无法识别的字符 '!'（是否想用 '!='？）")
            i += 1
        else:
            errors.append(f"第 {line} 行: 无法识别的字符 {ch!r}")
            i += 1
    tokens.append(Token("EOF", "", line))
    return tokens


class MiniCompiler:
    """递归下降解析 + 单遍语法制导翻译，语句级恐慌模式错误恢复。"""

    def __init__(self, source):
        self.errors = []
        self.tokens = tokenize(source, self.errors)
        self.pos = 0
        self.code = []          # IR 指令序列，元素为 tuple
        self.temp_id = 0
        self.label_id = 0
        self.symbols = set()    # 已声明变量（全局作用域）

    # ---------- 基础设施 ----------
    def error(self, line, msg):
        self.errors.append(f"第 {line} 行: {msg}")

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def expect(self, kind, what):
        tok = self.peek()
        if tok.kind != kind:
            got = "文件结尾" if tok.kind == "EOF" else repr(tok.value)
            self.error(tok.line, f"语法错误：期望 {what}，却得到 {got}")
            raise ParseError
        return self.advance()

    def emit(self, op, *args):
        self.code.append((op, *args))

    def new_temp(self):
        name = f"t{self.temp_id}"
        self.temp_id += 1
        return name

    def new_label(self):
        name = f"L{self.label_id}"
        self.label_id += 1
        return name

    def synchronize(self):
        """恐慌模式：丢弃 token 直到语句边界（';' 之后 / '}' 之前 / 语句关键字）。"""
        while self.peek().kind != "EOF":
            if self.tokens[self.pos - 1].kind == ";":
                return
            if self.peek().kind in ("}", "let", "if", "while"):
                return
            self.advance()

    # ---------- 入口 ----------
    def compile(self):
        while self.peek().kind != "EOF":
            try:
                self.statement()
            except ParseError:
                self.synchronize()
        return self.code

    # ---------- 语句 ----------
    def statement(self):
        tok = self.peek()
        if tok.kind == "let":
            self.decl()
        elif tok.kind == "if":
            self.if_stmt()
        elif tok.kind == "while":
            self.while_stmt()
        elif tok.kind == "{":
            self.block()
        elif tok.kind == "IDENT":
            self.assign()
        elif tok.kind == "}":
            self.error(tok.line, "意外的 '}'（没有匹配的 '{'）")
            self.advance()
        else:
            got = "文件结尾" if tok.kind == "EOF" else repr(tok.value)
            self.error(tok.line, f"无法识别的语句起始 {got}（期望 let/if/while/变量名/'{{'）")
            raise ParseError

    def decl(self):
        self.advance()  # 'let'
        name_tok = self.expect("IDENT", "变量名")
        redeclared = name_tok.value in self.symbols
        if redeclared:
            self.error(name_tok.line, f"变量 '{name_tok.value}' 重复声明")
        self.expect("=", "'='")
        src = self.expr()
        self.expect(";", "';'")
        if not redeclared:
            self.symbols.add(name_tok.value)  # 表达式求值后才可见，let x = x; 会报未声明
        self.emit("STORE", name_tok.value, src)

    def assign(self):
        name_tok = self.expect("IDENT", "变量名")
        if name_tok.value not in self.symbols:
            self.error(name_tok.line, f"变量 '{name_tok.value}' 未声明")
        self.expect("=", "'='")
        src = self.expr()
        self.expect(";", "';'")
        self.emit("STORE", name_tok.value, src)

    def if_stmt(self):
        self.advance()  # 'if'
        cond = self.expr()
        else_label = self.new_label()
        self.emit("JZ", cond, else_label)
        self.block()
        if self.peek().kind == "else":
            self.advance()
            end_label = self.new_label()
            self.emit("JMP", end_label)
            self.emit("LABEL", else_label)
            self.block()
            self.emit("LABEL", end_label)
        else:
            self.emit("LABEL", else_label)

    def while_stmt(self):
        self.advance()  # 'while'
        begin_label = self.new_label()
        end_label = self.new_label()
        self.emit("LABEL", begin_label)
        cond = self.expr()
        self.emit("JZ", cond, end_label)
        self.block()
        self.emit("JMP", begin_label)
        self.emit("LABEL", end_label)

    def block(self):
        self.expect("{", "'{'")
        while self.peek().kind not in ("}", "EOF"):
            try:
                self.statement()
            except ParseError:
                self.synchronize()
        self.expect("}", "'}'")

    # ---------- 表达式（按优先级分层，返回操作数：字面量/变量/临时变量） ----------
    def expr(self):
        return self.equality()

    def _binary_level(self, sub_level, ops):
        left = sub_level()
        while self.peek().kind in ops:
            op = BINOPS[self.advance().kind]
            right = sub_level()
            dst = self.new_temp()
            self.emit(op, dst, left, right)
            left = dst
        return left

    def equality(self):
        return self._binary_level(self.relational, ("==", "!="))

    def relational(self):
        return self._binary_level(self.additive, ("<", ">", "<=", ">="))

    def additive(self):
        return self._binary_level(self.multiplicative, ("+", "-"))

    def multiplicative(self):
        return self._binary_level(self.unary, ("*", "/", "%"))

    def unary(self):
        if self.peek().kind == "-":
            self.advance()
            operand = self.unary()
            dst = self.new_temp()
            self.emit("NEG", dst, operand)
            return dst
        return self.primary()

    def primary(self):
        tok = self.peek()
        if tok.kind == "INT":
            self.advance()
            return tok.value
        if tok.kind == "IDENT":
            self.advance()
            if tok.value not in self.symbols:
                self.error(tok.line, f"变量 '{tok.value}' 未声明")
            return tok.value
        if tok.kind == "(":
            self.advance()
            val = self.expr()
            self.expect(")", "')'")
            return val
        got = "文件结尾" if tok.kind == "EOF" else repr(tok.value)
        self.error(tok.line, f"表达式残缺：期望 数字/变量/'('，却得到 {got}")
        raise ParseError


def format_ir(code):
    """指令编号打印；LABEL 单独成行，体现嵌套配对结构。"""
    out, idx = [], 0
    for ins in code:
        if ins[0] == "LABEL":
            out.append(f"{ins[1]}:")
        else:
            out.append(f"{idx:3d}: {ins[0]} " + ", ".join(ins[1:]))
            idx += 1
    return "\n".join(out)


def run(source):
    compiler = MiniCompiler(source)
    code = compiler.compile()
    if compiler.errors:
        print("=== 错误报告 ===")
        for e in compiler.errors:
            print("  " + e)
        print(f"共 {len(compiler.errors)} 个错误，未生成中间表示。")
        return 1
    print("=== 中间表示(IR) ===")
    print(format_ir(code))
    print(f"（共 {sum(1 for i in code if i[0] != 'LABEL')} 条指令，无错误）")
    return 0


DEMO_OK = """\
# 正确示例：嵌套 if/while，演示标签“后开先关”配对
let n = 0;
let limit = 10;
while n < limit {
    if n % 2 == 0 {
        n = n + 2;
    } else {
        n = n + 1;
    }
}
"""

DEMO_BAD = """\
let x = 1;
let x = 2;
y = x * 3;
let z = ;
if x > 0
    x = x - 1;
}
while x < 5 {
    x = x + ;
}
"""


def main(argv):
    if len(argv) >= 2 and argv[1] == "--demo":
        print("########## 示例 1：正确程序 ##########")
        print(DEMO_OK)
        run(DEMO_OK)
        print()
        print("########## 示例 2：含各类错误的程序 ##########")
        print(DEMO_BAD)
        run(DEMO_BAD)
        return 0
    if len(argv) >= 2 and argv[1] != "-":
        with open(argv[1], encoding="utf-8") as f:
            source = f.read()
    else:
        source = sys.stdin.read()
    return run(source)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
