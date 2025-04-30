import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from typing_extensions import Annotated

from dialoscope.core.debate_manager import DebateManager
from dialoscope.core.agents import DEFAULT_MODEL

# 初始化 Typer app 和 Rich Console
app = typer.Typer(help="Dialoscope: AI 多角色辩论框架")
console = Console()

# 定义角色颜色
ROLE_STYLES = {
    "系统": "dim blue",
    "正方": "bold green",
    "反方": "bold red",
    "评审": "bold yellow",
    "评审 (最终报告)": "bold magenta",
    "错误": "bold white on red",
}

def print_message(role: str, content: str):
    """使用 Rich Console 打印带样式的消息，并渲染 Markdown"""
    style = ROLE_STYLES.get(role, "default")
    # 创建 Markdown 对象来渲染内容
    markdown_content = Markdown(content)
    # 将渲染后的 Markdown 内容放入 Panel 中
    console.print(Panel(markdown_content, title=role, border_style=style, expand=True))

@app.command()
def run(
    proposition: Annotated[str, typer.Argument(help="请输入辩论的命题或问题。例如：'人工智能最终会拥有自我意识吗？'")],
    rounds: Annotated[int, typer.Option("-r", "--rounds", help="设置辩论的轮数")] = 3,
    log_dir: Annotated[str, typer.Option("--log-dir", help="指定日志文件存放目录")] = "logs",
    model: Annotated[str, typer.Option("--model", help="指定使用的 LLM 模型")] = DEFAULT_MODEL,
):
    """开始一场新的 AI 辩论"""
    # 使用 Markdown 来格式化初始信息
    init_message = f"欢迎使用 Dialoscope!\n\n辩题: **{proposition}**\n轮数: {rounds}\n模型: {model}"
    console.print(Panel(Markdown(init_message), title="Dialoscope 初始化", border_style="blue"))

    try:
        manager = DebateManager(
            proposition=proposition,
            rounds=rounds,
            log_dir=log_dir,
            model=model,
            printer=print_message
        )
        manager.run_debate()
        console.print(Panel(Markdown("**辩论流程结束。**"), border_style="green"))
    except (ValueError, RuntimeError) as e:
        print_message("错误", f"初始化失败: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print_message("错误", f"运行过程中出现意外错误: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
