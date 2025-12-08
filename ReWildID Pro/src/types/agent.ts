// Agent-related types for the chat interface

export interface CodeExecutionResult {
    success: boolean;
    error: string | null;
    output: string | null;
    images: string[];
    code: string;
}

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'tool';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
    toolCalls?: ToolCallResult[];
    images?: string[]; // Base64 data URLs for code execution results
    codeExecution?: CodeExecutionResult; // Parsed code execution result
}

export interface ToolCallResult {
    toolName: string;
    args: Record<string, unknown>;
    result: string;
}

export interface AgentSession {
    id: string;
    title: string;
    messages: ChatMessage[];
    createdAt: Date;
    updatedAt: Date;
}

export interface AgentSettings {
    apiKey: string;
    model: string;
    e2bApiKey: string;
}

export const DEFAULT_AGENT_SETTINGS: AgentSettings = {
    apiKey: '',
    model: 'gemini-flash-latest',
    e2bApiKey: '',
};
