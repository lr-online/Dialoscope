import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from typing_extensions import Annotated
import sys

from dialoscope.core.debate_manager import DebateManager
# Import necessary components for config and LLM clients
from dialoscope.config import get_provider_config, CONFIG # Import CONFIG too for default
from dialoscope.llm_clients import OpenAIClient, OllamaClient, LLMClient
from dialoscope.core.agents import ProponentAgent, OpponentAgent, JudgeAgent

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
    # Changed --model to --provider
    provider: Annotated[str, typer.Option("--provider", help="选择 LLM 提供商 (例如 'openai', 'ollama')")] = CONFIG.get('default_provider', 'openai'), # Default from config
):
    """开始一场新的 AI 辩论"""
    
    # Initialize LLM client based on provider
    llm_client: LLMClient
    selected_provider = provider.lower()
    try:
        provider_config = get_provider_config(selected_provider)
        
        # Display chosen provider and model for clarity
        model_name = provider_config.get('model', '(未指定)')
        init_message = f"欢迎使用 Dialoscope!\n\n辩题: **{proposition}**\n轮数: {rounds}\nLLM 提供商: {selected_provider}\n模型: {model_name}" # Show model from config
        console.print(Panel(Markdown(init_message), title="Dialoscope 初始化", border_style="blue"))

        if selected_provider == 'openai':
            llm_client = OpenAIClient(provider_config)
            print_message("系统", f"使用 OpenAI 客户端 (模型: {model_name})")
        elif selected_provider == 'ollama':
            llm_client = OllamaClient(provider_config)
            print_message("系统", f"使用 Ollama 客户端 (模型: {model_name}, URL: {provider_config.get('base_url')})")
        else:
            print_message("错误", f"不支持的 LLM 提供商: '{selected_provider}'。请选择 'openai' 或 'ollama'。")
            raise typer.Exit(code=1)
            
    except FileNotFoundError as e:
         print_message("错误", f"无法找到配置文件 config.yaml: {e}")
         raise typer.Exit(code=1)
    except ValueError as e:
         print_message("错误", f"配置错误: {e}")
         raise typer.Exit(code=1)
    except Exception as e: # Catch potential client initialization errors
         print_message("错误", f"初始化 LLM 客户端 '{selected_provider}' 时出错: {e}")
         raise typer.Exit(code=1)

    # Initialize agents with the created client
    try:
        proponent = ProponentAgent(llm_client=llm_client)
        opponent = OpponentAgent(llm_client=llm_client)
        judge = JudgeAgent(llm_client=llm_client)
    except Exception as e:
        print_message("错误", f"初始化 Agent 时出错: {e}")
        raise typer.Exit(code=1)

    # Initialize and run the debate manager
    try:
        manager = DebateManager(
            proposition=proposition,
            proponent=proponent, # Pass agent instances
            opponent=opponent,
            judge=judge,
            rounds=rounds,
            log_dir=log_dir,
            printer=print_message
            # Removed model parameter
        )
        manager.run_debate()
        console.print(Panel(Markdown("**辩论流程结束。**"), border_style="green"))
    # Catch errors during debate run (already handled inside manager.run_debate, but keep a top-level catch)
    except Exception as e:
        print_message("错误", f"辩论运行过程中出现意外错误: {e}")
        # Optionally log the full traceback here for debugging
        # import traceback
        # print_message("错误", f"Traceback:\n{traceback.format_exc()}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
