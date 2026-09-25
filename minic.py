#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minic.py — 迷你语言编译器：源脚本 -> 三地址码中间表示（IR）
纯 Python 标准库，单文件，无第三方依赖。

用法：
    python3 minic.py 文件路径        # 编译指定源文件
    python3 minic.py < 文件路径      # 从标准输入读入
    python3 minic.py --demo          # 运行内置正确示例
    python3 minic.py --demo-err      # 运行内置错误示例

语言语法（EBNF）：
    program := stmt*
    stmt    := "let" IDENT "=" expr ";"            # 声明并初始化
             | IDENT "=" expr ";"                  # 赋值（变量须已声明）
             | "if" expr block ("else" block)?     # 条件语句
             | "while" expr block                  # 循环语句
    block   := "{" stmt* "}"
    expr    := cmp
    cmp     := add (("=="|"!="|"<"|"<="|">"|">=") add)?
    add     := mul (("+"|"-") mul)*
    mul     := unary (("*"|"/"|"%") unary)*
    unary   := "-" unary | primary
    primary := NUMBER | IDENT | "(" expr ")"
    注释：# 到行尾

IR 设计说明（三类指令，三地址码形式）：
  1. 运算类：CONST / LOAD / NEG / ADD SUB MUL DIV MOD / CMP_EQ..CMP_GE
     —— 每条指令至多一个运算、三个操作数，表达式按优先级被拆成
        与源语序解耦的线性序列，便于后续优化（常量折叠、公共子表达式
        消除）与向任意后端（栈机/寄存器机）翻译。
  2. 存储类：STORE
     —— 全部变量写操作收敛到这一条指令，读操作只有 LOAD，定义-使用
        关系一目了然，便于做数据流分析；变量必须先 let 声明才能使用。
  3. 跳转类：LABEL / JMP / JZ（条件为假跳转）
     —— 结构化的 if/while 全部降为这三条原语；跳转目标配对（嵌套时
        “后开先关”）由代码生成的递归结构天然保证：每进入一层 if/while
        就新申请一对标签，递归返回前先把本层标签闭合。
"""

import sys
from dataclasses import dataclass

# ---------------------------------------------------------------- 词法分析

KEYWORDS = {"let": "LET", "if": "IF", "else": "ELSE", "while": "WHILE"}
TWO_CHAR_OPS = ("==", "!=", "<=", ">=")
ONE_CHAR_TOKENS = set("+-*/%<>=(){};")


@dataclass
class Token:
    kind: str   # NUMBER / IDENT / LET / IF / ELSE / WHILE / 符号本身 / EOF
    value: object
    line: int


def tokenize(src, errors):
    """把源文本切成 token 序列；非法字符记入 errors 并跳过。"""
    toks = []
    i, line, n = 0, 1, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r":
            i += 1
        elif c == "\n":
            line += 1
            i += 1
        elif c == "#":  # 注释
            while i < n and src[i] != "\n":
                i += 1
        elif c.isdigit():
            j = i
            while j < n and src[j].isdigit():
                j += 1
            toks.append(Token("NUMBER", int(src[i:j]), line))
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            toks.append(Token(KEYWORDS.get(word, "IDENT"), word, line))
            i = j
        elif src.startswith(TWO_CHAR_OPS, i):
            toks.append(Token(src[i:i + 2], src[i:i + 2], line))
            i += 2
        elif c in ONE_CHAR_TOKENS:
            toks.append(Token(c, c, line))
            i += 1
        else:
            errors.append((line, f"无法识别的字符 {c!r}"))
            i += 1
    toks.append(Token("EOF", "", line))
    return toks


# ---------------------------------------------------------------- 语法树

@dataclass
class Decl:    # let name = init;
    name: str
    init: object
    line: int


@dataclass
class Assign:  # name = value;
    name: str
    value: object
    line: int


@dataclass
class If:      # if cond { then } [else { otherwise }]
    cond: object
    then_body: list
    else_body: list
    line: int


@dataclass
class While:   # while cond { body }
    cond: object
    body: list
    line: int


@dataclass
class Num:
    value: int
    line: int


@dataclass
class Var:
    name: str
    line: int


@dataclass
class Bin:     # 二元运算
    op: str
    left: object
    right: object
    line: int


@dataclass
class Neg:     # 一元负号
    operand: object
    line: int


# ---------------------------------------------------------------- 语法分析

class ParseError(Exception):
    def __init__(self, line, msg):
        super().__init__(msg)
        self.line = line
        self.msg = msg


class Parser:
    """递归下降 + 优先级分层；出错时恐慌模式恢复到下一语句边界。"""

    def __init__(self, toks, errors):
        self.toks = toks
        self.pos = 0
        self.errors = errors

    def peek(self):
        return self.toks[min(self.pos, len(self.toks) - 1)]

    def advance(self):
        tok = self.toks[min(self.pos, len(self.toks) - 1)]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def expect(self, kind, desc):
        tok = self.peek()
        if tok.kind != kind:
            got = "文件结尾" if tok.kind == "EOF" else f"'{tok.value}'"
            raise ParseError(tok.line, f"缺少{desc}：期望 '{kind}'，实际遇到 {got}")
        return self.advance()

    def synchronize(self):
        """恐慌模式恢复：跳到下一个 ';'（吃掉）或 '}'（留给外层）。"""
        while self.peek().kind not in ("EOF", "}"):
            if self.advance().kind == ";":
                return

    def parse_program(self):
        stmts = []
        while self.peek().kind != "EOF":
            if self.peek().kind == "}":  # 防止顶层多余 '}' 造成死循环
                tok = self.advance()
                self.errors.append((tok.line, "多余的 '}'：没有匹配的 '{'"))
                continue
            try:
                stmts.append(self.parse_stmt())
            except ParseError as e:
                self.errors.append((e.line, e.msg))
                self.synchronize()
        return stmts

    def parse_stmt(self):
        tok = self.peek()
        if tok.kind == "LET":
            self.advance()
            name = self.expect("IDENT", "变量名")
            self.expect("=", "'='")
            init = self.parse_expr()
            self.expect(";", "';'")
            return Decl(name.value, init, tok.line)
        if tok.kind == "IDENT":
            self.advance()
            self.expect("=", "'='（赋值语句）")
            value = self.parse_expr()
            self.expect(";", "';'")
            return Assign(tok.value, value, tok.line)
        if tok.kind == "IF":
            self.advance()
            cond = self.parse_expr()
            then_body = self.parse_block()
            else_body = []
            if self.peek().kind == "ELSE":
                self.advance()
                else_body = self.parse_block()
            return If(cond, then_body, else_body, tok.line)
        if tok.kind == "WHILE":
            self.advance()
            cond = self.parse_expr()
            body = self.parse_block()
            return While(cond, body, tok.line)
        got = "文件结尾" if tok.kind == "EOF" else f"'{tok.value}'"
        raise ParseError(tok.line,
                         f"无法识别的语句开头 {got}（应为 let / if / while / 赋值）")

    def parse_block(self):
        self.expect("{", "'{'")
        stmts = []
        while self.peek().kind not in ("}", "EOF"):
            try:
                stmts.append(self.parse_stmt())
            except ParseError as e:
                self.errors.append((e.line, e.msg))
                self.synchronize()
        self.expect("}", "'}'")
        return stmts

    # 表达式：优先级从低到高分层 —— 比较 < 加减 < 乘除模 < 负号 < 原子
    def parse_expr(self):
        return self.parse_cmp()

    def parse_cmp(self):
        left = self.parse_add()
        tok = self.peek()
        if tok.kind in ("==", "!=", "<", "<=", ">", ">="):
            self.advance()
            right = self.parse_add()
            return Bin(tok.kind, left, right, tok.line)
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.peek().kind in ("+", "-"):
            op = self.advance()
            left = Bin(op.kind, left, self.parse_mul(), op.line)
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.peek().kind in ("*", "/", "%"):
            op = self.advance()
            left = Bin(op.kind, left, self.parse_unary(), op.line)
        return left

    def parse_unary(self):
        tok = self.peek()
        if tok.kind == "-":
            self.advance()
            return Neg(self.parse_unary(), tok.line)
        return self.parse_primary()

    def parse_primary(self):
        tok = self.peek()
        if tok.kind == "NUMBER":
            self.advance()
            return Num(tok.value, tok.line)
        if tok.kind == "IDENT":
            self.advance()
            return Var(tok.value, tok.line)
        if tok.kind == "(":
            self.advance()
            e = self.parse_expr()
            self.expect(")", "')'")
            return e
        got = "文件结尾" if tok.kind == "EOF" else f"'{tok.value}'"
        raise ParseError(tok.line, f"表达式残缺：在 {got} 处缺少操作数")


# ---------------------------------------------------------------- 代码生成

BIN_OPS = {
    "+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "%": "MOD",
    "==": "CMP_EQ", "!=": "CMP_NE", "<": "CMP_LT", "<=": "CMP_LE",
    ">": "CMP_GT", ">=": "CMP_GE",
}


class CodeGen:
    """AST -> 三地址码；同时做声明检查（未声明使用 / 重复声明）。"""

    def __init__(self):
        self.code = []
        self.errors = []
        self.symbols = set()  # 已声明变量
        self.n_temp = 0
        self.n_label = 0

    def new_temp(self):
        t = f"t{self.n_temp}"
        self.n_temp += 1
        return t

    def new_label(self):
        lb = f"L{self.n_label}"
        self.n_label += 1
        return lb

    def emit(self, *ins):
        self.code.append(ins)

    def gen_program(self, stmts):
        for s in stmts:
            self.gen_stmt(s)

    def gen_stmt(self, s):
        if isinstance(s, Decl):
            if s.name in self.symbols:
                self.errors.append((s.line, f"变量 '{s.name}' 重复声明"))
            else:
                self.symbols.add(s.name)
            v = self.gen_expr(s.init)
            self.emit("STORE", s.name, v)
        elif isinstance(s, Assign):
            if s.name not in self.symbols:
                self.errors.append((s.line, f"变量 '{s.name}' 未声明就使用"))
            v = self.gen_expr(s.value)
            self.emit("STORE", s.name, v)
        elif isinstance(s, If):
            # JZ 到 else，JMP 到 end —— 标签成对申请、递归返回前闭合
            c = self.gen_expr(s.cond)
            l_else, l_end = self.new_label(), self.new_label()
            self.emit("JZ", c, l_else)
            for st in s.then_body:
                self.gen_stmt(st)
            self.emit("JMP", l_end)
            self.emit("LABEL", l_else)
            for st in s.else_body:
                self.gen_stmt(st)
            self.emit("LABEL", l_end)
        elif isinstance(s, While):
            # Lbegin: 条件; JZ 到 Lend; 循环体; JMP 回 Lbegin; Lend:
            l_begin, l_end = self.new_label(), self.new_label()
            self.emit("LABEL", l_begin)
            c = self.gen_expr(s.cond)
            self.emit("JZ", c, l_end)
            for st in s.body:
                self.gen_stmt(st)
            self.emit("JMP", l_begin)
            self.emit("LABEL", l_end)

    def gen_expr(self, e):
        if isinstance(e, Num):
            t = self.new_temp()
            self.emit("CONST", t, e.value)
            return t
        if isinstance(e, Var):
            if e.name not in self.symbols:
                self.errors.append((e.line, f"变量 '{e.name}' 未声明就使用"))
            t = self.new_temp()
            self.emit("LOAD", t, e.name)
            return t
        if isinstance(e, Neg):
            a = self.gen_expr(e.operand)
            t = self.new_temp()
            self.emit("NEG", t, a)
            return t
        if isinstance(e, Bin):
            left = self.gen_expr(e.left)
            right = self.gen_expr(e.right)
            t = self.new_temp()
            self.emit(BIN_OPS[e.op], t, left, right)
            return t
        raise AssertionError(f"未知表达式节点: {e!r}")


# ---------------------------------------------------------------- 驱动

def compile_source(src):
    """返回 (指令序列, [(行号, 错误信息), ...])，错误按行号排序。"""
    errors = []
    toks = tokenize(src, errors)
    program = Parser(toks, errors).parse_program()
    gen = CodeGen()
    gen.gen_program(program)
    errors.extend(gen.errors)
    errors.sort(key=lambda e: e[0])
    return gen.code, errors


def format_ir(code):
    out = []
    for i, ins in enumerate(code):
        operands = ", ".join(str(a) for a in ins[1:])
        out.append(f"{i:>4}  {ins[0]:<7} {operands}".rstrip())
    return "\n".join(out)


DEMO_SRC = """\
# 正确示例：声明、运算优先级、if/else 与 while 嵌套
let a = 10;
let b = 3;
let c = a + b * 2;      # 先乘后加
if c > 15 {
    while c > 0 {       # 循环嵌套在条件里，标签后开先关
        c = c - 1;
    }
} else {
    c = 0;
}
"""

DEMO_ERR_SRC = """\
let x = 1;
let x = 2;          # 重复声明
y = x + 1;          # y 未声明
let z = ;           # 表达式残缺
if x > 0            # 缺少 '{'
    x = x - 1;
while x > 0 {
    x = x - ;       # 表达式残缺（右操作数缺失）
"""


def main(argv):
    if len(argv) > 1 and argv[1] == "--demo":
        src = DEMO_SRC
        print("== 输入源代码 ==")
        print(src)
    elif len(argv) > 1 and argv[1] == "--demo-err":
        src = DEMO_ERR_SRC
        print("== 输入源代码 ==")
        print(src)
    elif len(argv) > 1:
        with open(argv[1], encoding="utf-8") as f:
            src = f.read()
    else:
        src = sys.stdin.read()

    code, errors = compile_source(src)

    print("== 中间表示（IR）==")
    print(format_ir(code) if code else "（无）")
    print()
    print("== 错误报告 ==")
    if errors:
        for line, msg in errors:
            print(f"第 {line} 行: {msg}")
        print(f"共 {len(errors)} 个错误")
        return 1
    print("无错误，编译成功")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
