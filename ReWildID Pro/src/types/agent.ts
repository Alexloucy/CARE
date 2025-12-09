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
    backupPath?: string;  // For updates - path to backup for reverting
}

// Confirmation request for destructive operations
export interface ConfirmationRequest {
    id: string;
    action: 'update_metadata' | 'delete_rows' | 'other';
    description: string;
    affectedCount: number;
    preview: string[];  // First few affected items for preview
    pendingCode: string;  // Python code to execute if confirmed
    filterSql: string;  // SQL WHERE clause for filtering affected rows
    status: 'pending' | 'applied' | 'reverted';  // Track state
    backupPath?: string;  // Path to backup file for reverting
}

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'tool' | 'confirmation';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;

    // For role='assistant' - tool calls the AI requested
    toolCalls?: ToolCall[];

    // For role='tool' - result from tool execution
    toolCallId?: string;
    toolName?: string;
    toolResult?: ToolResult;

    // For role='confirmation' - pending action requiring user approval
    confirmationRequest?: ConfirmationRequest;
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

// Available AI models for the agent (December 2025)
export const AVAILABLE_MODELS = [
    { id: 'gemini-flash-latest', name: 'Gemini Flash', description: 'Hybrid reasoning, 1M context' },
    { id: 'gemini-flash-lite-latest', name: 'Gemini Flash Lite', description: 'Fastest and cheapest' },
    { id: 'gemini-3-pro-preview', name: 'Gemini 3 Pro', description: 'Most intelligent (preview)' },
] as const;

export const DEFAULT_AGENT_SETTINGS: AgentSettings = {
    apiKey: '',
    model: 'gemini-flash-latest',
};
