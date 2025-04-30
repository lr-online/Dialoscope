import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from typing_extensions import Annotated
from typing import Iterator, Dict

from dialoscope.core.debate_manager import DebateManager
from dialoscope.config import get_agent_config
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

# Modified print_message to handle streaming with Rich Live
def print_stream(role: str, content_stream: Iterator[str]) -> str:
    """Uses Rich Live to print a stream of content chunks into a Panel with Markdown rendering.

    Args:
        role: The role of the speaker.
        content_stream: An iterator yielding string chunks of the content.

    Returns:
        The fully accumulated content string.
    """
    style = ROLE_STYLES.get(role, "default")
    # Initialize panel with an empty Markdown object to start
    panel = Panel(Markdown(""), title=role, border_style=style, expand=True)
    accumulated_content = ""

    with Live(panel, console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        for chunk in content_stream:
            accumulated_content += chunk
            # Update the panel's renderable with a new Markdown object containing the accumulated content
            panel.renderable = Markdown(accumulated_content)
            live.update(panel) # Refresh the display with the updated panel (containing Markdown)

        # Final update to ensure the last chunk is rendered (optional, but good practice)
        # panel.renderable = Markdown(accumulated_content)
        # live.update(panel)
        
    return accumulated_content

# Keep the old print_message for non-streaming system messages
def print_message(role: str, content: str):
    """Prints a complete message using Rich Panel and Markdown."""
    style = ROLE_STYLES.get(role, "default")
    markdown_content = Markdown(content)
    console.print(Panel(markdown_content, title=role, border_style=style, expand=True))

@app.command()
def run(
    proposition: Annotated[str, typer.Argument(help="请输入辩论的命题或问题。例如：'人工智能最终会拥有自我意识吗？'")],
    rounds: Annotated[int, typer.Option("-r", "--rounds", help="设置辩论的轮数")] = 3,
    log_dir: Annotated[str, typer.Option("--log-dir", help="指定日志文件存放目录")] = "logs",
    # 移除 model 选项，因为我们将依赖 config 文件进行 agent 级别的模型配置
    # model: Annotated[Optional[str], typer.Option("--model", help="指定要使用的具体 LLM 模型...")] = None,
):
    """开始一场新的 AI 辩论"""
    
    llm_clients: Dict[str, LLMClient] = {} # Store clients per agent role
    agent_roles = ["proponent", "opponent", "judge"] # Define the roles

    try:

        init_message = f"欢迎使用 Dialoscope!\n\n辩题: **{proposition}**\n轮数: {rounds}\n"
        console.print(Panel(Markdown(init_message), title="Dialoscope 初始化", border_style="blue"))

        # Create a separate client for each agent role
        for role in agent_roles:
            agent_config = get_agent_config(role)
            agent_model = agent_config.get('model') # Already validated by get_agent_config

            # Instantiate the appropriate client based on the provider
            client_instance: LLMClient
            # Get the provider name *from the agent's config* now
            agent_provider = agent_config.get('provider')
            if not agent_provider:
                 # This should ideally be caught by get_agent_config already, but double-check
                 print_message("错误", f"Agent '{role}' 的配置中缺少 'provider'。")
                 raise typer.Exit(code=1)
                 
            if agent_provider == 'openai':
                client_instance = OpenAIClient(agent_config)
            elif agent_provider == 'ollama':
                client_instance = OllamaClient(agent_config)
            # Add other providers here if needed
            # elif agent_provider == 'anthropic':
            #     client_instance = AnthropicClient(agent_config)
            else:
                print_message("错误", f"不支持的 LLM 提供商: '{agent_provider}' (在 agent '{role}' 的配置中指定)。请在 cli.py 中添加支持。")
                raise typer.Exit(code=1)
            
            llm_clients[role] = client_instance
            
    except FileNotFoundError as e:
         print_message("错误", f"无法找到配置文件 config.yaml: {e}")
         raise typer.Exit(code=1)
    except ValueError as e:
         print_message("错误", f"配置错误: {e}")
         raise typer.Exit(code=1)
    except Exception as e: # Catch potential client/config initialization errors
         print_message("错误", f"初始化 LLM 客户端时出错: {e}")
         # import traceback
         # print_message("错误", f"Traceback:\n{traceback.format_exc()}")
         raise typer.Exit(code=1)

    # Initialize agents with their specific clients
    try:
        # Ensure all required clients were created
        if not all(role in llm_clients for role in agent_roles):
            missing = [r for r in agent_roles if r not in llm_clients]
            print_message("错误", f"未能为所有 agent 角色创建 LLM 客户端。缺少: {missing}")
            raise typer.Exit(code=1)
            
        proponent = ProponentAgent(llm_client=llm_clients["proponent"])
        opponent = OpponentAgent(llm_client=llm_clients["opponent"])
        judge = JudgeAgent(llm_client=llm_clients["judge"])
    except Exception as e:
        print_message("错误", f"初始化 Agent 时出错: {e}")
        raise typer.Exit(code=1)

    # Initialize and run the debate manager
    try:
        manager = DebateManager(
            proposition=proposition,
            proponent=proponent,
            opponent=opponent,
            judge=judge,
            rounds=rounds,
            log_dir=log_dir,
            printer=print_message,
            stream_printer=print_stream
        )
        manager.run_debate()
        console.print(Panel(Markdown("**辩论流程结束。**"), border_style="green"))
    except Exception as e:
        print_message("错误", f"辩论运行过程中出现意外错误: {e}")
        # import traceback
        # print_message("错误", f"Traceback:\n{traceback.format_exc()}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
