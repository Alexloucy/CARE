// Agent-related types for the chat interface

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'tool';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
    toolCalls?: ToolCallResult[];
}

export interface ToolCallResult {
    toolName: string;
    args: Record<string, unknown>;
    result: string;
}

export interface AgentSession {
    id: string;
    messages: ChatMessage[];
    createdAt: Date;
    updatedAt: Date;
}

export interface AgentSettings {
    apiKey: string;
    model: string;
}

export const DEFAULT_AGENT_SETTINGS: AgentSettings = {
    apiKey: '',
    model: 'gemini-flash-latest',
};
