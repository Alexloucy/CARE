// Agent-related types for the chat interface

// Tool call that agent requested
export interface ToolCall {
    id: string;
    name: string;
    args: Record<string, unknown>;
}

// Result from tool execution
export interface ToolResult {
    success: boolean;
    output: string | null;
    error: string | null;
    images?: string[];  // Base64 data URLs for generated images
    code?: string;      // For code execution - the code that was run
}

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'tool';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;

    // For role='assistant' - tool calls the AI requested
    toolCalls?: ToolCall[];

    // For role='tool' - result from tool execution
    toolCallId?: string;
    toolName?: string;
    toolResult?: ToolResult;
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
}

export const DEFAULT_AGENT_SETTINGS: AgentSettings = {
    apiKey: '',
    model: 'gemini-2.0-flash',
};
