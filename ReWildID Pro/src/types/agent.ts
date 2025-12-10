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
    action: 'update_metadata' | 'delete_rows' | 'run_classification' | 'run_reid' | 'other';
    description: string;
    affectedCount: number;
    preview: string[];  // First few affected items for preview
    pendingCode?: string;  // Python code to execute if confirmed (for metadata updates)
    filterSql?: string;  // SQL WHERE clause for filtering affected rows
    status: 'pending' | 'applied' | 'reverted';  // Track state
    backupPath?: string;  // Path to backup file for reverting
    // Job-specific data
    imageIds?: number[];  // Image IDs for classification/reid jobs
    species?: string;  // Species for ReID job
}

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'tool' | 'confirmation';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;
    images?: string[];  // Base64 data URLs for attached images (user messages)

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
    enabled: boolean;  // Whether AI Agent is enabled (shown in sidebar)
    hasAgreedToTerms: boolean;  // Whether user has agreed to data privacy terms
    // Image generation settings
    imageGenerationModel: string;  // 'gemini-2.5-flash-image' or 'gemini-3-pro-image-preview'
    imageResolution: string;  // '1K', '2K', '4K' (only for Pro model)
}

// Available AI models for the agent (December 2025)
export const AVAILABLE_MODELS = [
    { id: 'gemini-flash-latest', name: 'Gemini Flash', description: 'Hybrid reasoning, 1M context' },
    { id: 'gemini-flash-lite-latest', name: 'Gemini Flash Lite', description: 'Fastest and cheapest' },
    { id: 'gemini-3-pro-preview', name: 'Gemini 3 Pro', description: 'Most intelligent (preview)' },
] as const;

// Available image generation models (Nano Banana / Nano Banana Pro)
export const IMAGE_GENERATION_MODELS = [
    { id: 'gemini-2.5-flash-image', name: 'Nano Banana', description: 'Fast, 1K resolution' },
    { id: 'gemini-3-pro-image-preview', name: 'Nano Banana Pro', description: 'Advanced, up to 4K, thinking mode' },
] as const;

// Available resolutions (only 2K/4K for Pro model)
export const IMAGE_RESOLUTIONS = [
    { id: '1K', name: '1K', description: '~1024px' },
    { id: '2K', name: '2K', description: '~2048px (Pro only)' },
    { id: '4K', name: '4K', description: '~4096px (Pro only)' },
] as const;

export const DEFAULT_AGENT_SETTINGS: AgentSettings = {
    apiKey: '',
    model: 'gemini-flash-latest',
    enabled: false,
    hasAgreedToTerms: false,
    imageGenerationModel: 'gemini-2.5-flash-image',
    imageResolution: '1K',
};

