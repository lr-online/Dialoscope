import argparse
from dialoscope.core.debate_manager import DebateManager

def main():
    parser = argparse.ArgumentParser(description="Dialoscope: AI 多角色辩论框架")
    parser.add_argument("proposition", type=str, help="请输入辩论的命题或问题")
    parser.add_argument("-r", "--rounds", type=int, default=3, help="设置辩论的轮数 (默认: 3)")
    parser.add_argument("--log-dir", type=str, default="logs", help="指定日志文件存放目录 (默认: logs)")
    # TODO: 后续可以添加选择 LLM 模型、配置 API Key 等参数

    args = parser.parse_args()

    if not args.proposition:
        print("错误：必须提供辩论命题。")
        parser.print_help()
        return

    print("欢迎使用 Dialoscope!")
    print("正在初始化辩论...")

    try:
        manager = DebateManager(
            proposition=args.proposition,
            rounds=args.rounds,
            log_dir=args.log_dir
        )
        manager.run_debate()
        print("\n辩论流程结束。")
    except Exception as e:
        print(f"\n运行过程中出现错误: {e}")

if __name__ == "__main__":
    main()
