sequenceDiagram
    participant User as 用户
    participant Session as ChatSession<br/>(控制器)
    participant Memory as ConversationMemory<br/>(记忆管理器)
    participant LLM as LLM API

    User->>Session: 1. 发送消息 "我叫张三"
    Session->>Memory: 2. add_user_message("我叫张三")
    activate Memory
    Memory->>Memory: 3. 追加至内部列表<br/>[{role: "user", content: "我叫张三"}]
    Memory->>Memory: 4. 计算当前列表总 Token 数
    alt Token 总数 > max_tokens
        Memory->>Memory: 5. 从最早的消息开始删除
    end
    deactivate Memory

    Session->>Memory: 6. get_messages()
    Memory-->>Session: 7. 返回完整历史列表

    Session->>LLM: 8. 调用 API (messages=历史)
    activate LLM
    LLM-->>Session: 9. 返回回复 "你好张三"
    deactivate LLM

    Session->>Memory: 10. add_assistant_message("你好张三")
    activate Memory
    Memory->>Memory: 11. 追加至内部列表
    deactivate Memory

    Session-->>User: 12. 显示回复 "你好张三"