// LangChain Agent Service for Google AI Studio (Gemini)
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { ChatMessage, AgentSettings, DEFAULT_AGENT_SETTINGS, ToolCall, ToolResult, AgentSession } from '../types/agent';

// Define the secret reveal tool
const revealSecretTool = tool(
    async () => {
        return JSON.stringify({
            success: true,
            output: 'The secret is: asoidfjaiosdfj',
            error: null,
        });
    },
    {
        name: 'revealSecret',
        description: 'Reveals a secret message when the user asks for it',
        schema: z.object({}),
    }
);

// Define the Python code execution tool (LOCAL execution via Electron)
const runPythonCodeTool = tool(
    async ({ code }: { code: string }) => {
        console.log('[Python] Starting local code execution, code length:', code.length);

        try {
            // Call the Electron main process to execute Python locally
            const result = await (window as any).api.executePythonCode(code);
            console.log('[Python] Execution complete:', result);

            return JSON.stringify({
                success: result.success,
                error: result.error,
                output: result.output,
                images: result.images || [],
                code: code,
            });
        } catch (error) {
            console.error('[Python] Exception:', error);
            return JSON.stringify({
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                output: null,
                images: [],
                code: code,
            });
        }
    },
    {
        name: 'runPythonCode',
        description: 'Execute Python code locally on the user\'s machine. Use this for calculations, data processing, and generating visualizations with matplotlib/seaborn. Code output and any generated images will be returned.',
        schema: z.object({
            code: z.string().describe('The Python code to execute. Include necessary imports like "import matplotlib.pyplot as plt". Use plt.show() to display charts.'),
        }),
    }
);

// Get the tools list
function getAvailableTools() {
    return [revealSecretTool, runPythonCodeTool as any];
}

// Storage keys
const SETTINGS_KEY = 'agent_settings';
const SESSIONS_KEY = 'agent_sessions';
const CURRENT_SESSION_KEY = 'agent_current_session';

// Session management
export function getSessions(): AgentSession[] {
    try {
        const stored = localStorage.getItem(SESSIONS_KEY);
        if (stored) {
            const sessions = JSON.parse(stored);
            return sessions.map((s: any) => ({
                ...s,
                createdAt: new Date(s.createdAt),
                updatedAt: new Date(s.updatedAt),
                messages: s.messages.map((m: any) => ({
                    ...m,
                    timestamp: new Date(m.timestamp),
                })),
            }));
        }
    } catch (e) {
        console.error('Failed to parse sessions:', e);
    }
    return [];
}

export function saveSession(session: AgentSession): void {
    const sessions = getSessions();
    const existingIdx = sessions.findIndex(s => s.id === session.id);
    if (existingIdx >= 0) {
        sessions[existingIdx] = session;
    } else {
        sessions.unshift(session);
    }
    const trimmed = sessions.slice(0, 50);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(trimmed));
}

export function deleteSession(sessionId: string): void {
    const sessions = getSessions().filter(s => s.id !== sessionId);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
}

export function getCurrentSessionId(): string | null {
    return localStorage.getItem(CURRENT_SESSION_KEY);
}

export function setCurrentSessionId(sessionId: string): void {
    localStorage.setItem(CURRENT_SESSION_KEY, sessionId);
}

// Get settings from localStorage
export function getAgentSettings(): AgentSettings {
    try {
        const stored = localStorage.getItem(SETTINGS_KEY);
        if (stored) {
            return { ...DEFAULT_AGENT_SETTINGS, ...JSON.parse(stored) };
        }
    } catch (e) {
        console.error('Failed to parse agent settings:', e);
    }
    return DEFAULT_AGENT_SETTINGS;
}

// Save settings to localStorage
export function saveAgentSettings(settings: Partial<AgentSettings>): void {
    const current = getAgentSettings();
    const updated = { ...current, ...settings };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(updated));
}

// Create the LangChain model instance
function createModel(settings: AgentSettings): ChatGoogleGenerativeAI | null {
    if (!settings.apiKey) {
        return null;
    }

    return new ChatGoogleGenerativeAI({
        apiKey: settings.apiKey,
        model: settings.model || 'gemini-2.0-flash',
        maxOutputTokens: 8192,
        temperature: 0.7,
    });
}

// Convert our messages to LangChain format
function toLangChainMessages(messages: ChatMessage[]): BaseMessage[] {
    const result: BaseMessage[] = [];

    for (const m of messages) {
        if (m.role === 'user') {
            result.push(new HumanMessage(m.content));
        } else if (m.role === 'tool') {
            // Tool result message
            if (m.toolCallId) {
                result.push(new ToolMessage({
                    tool_call_id: m.toolCallId,
                    content: JSON.stringify(m.toolResult || { success: true, output: m.content }),
                }));
            }
        } else if (m.role === 'assistant') {
            // Assistant message - may have tool calls
            if (m.toolCalls && m.toolCalls.length > 0) {
                result.push(new AIMessage({
                    content: m.content || '',
                    tool_calls: m.toolCalls.map(tc => ({
                        id: tc.id,
                        name: tc.name,
                        args: tc.args,
                    })),
                }));
            } else {
                result.push(new AIMessage(m.content));
            }
        }
    }

    return result;
}

// System prompt for the agent
const SYSTEM_PROMPT = `You are a helpful AI assistant for RewildID Pro, a wildlife re-identification application. 
You help users with wildlife conservation tasks, image analysis, and general questions.

Available tools:
- revealSecret: Reveals a secret message when users ask for it
- runPythonCode: Execute Python code to perform calculations, data analysis, or generate charts/visualizations with matplotlib. Use this tool when users want to see graphs, charts, plots, or need computational tasks done.

When generating visualizations:
1. Always use matplotlib.pyplot
2. Use plt.show() to display charts - images are automatically captured
3. For nice charts, consider using seaborn
4. Add proper titles, labels, and legends

Be friendly, concise, and helpful. When users ask for data visualization or charts, proactively use the runPythonCode tool.`;

// Stream chunk types
export type StreamChunk =
    | { type: 'text'; content: string }
    | { type: 'tool_call'; toolCall: ToolCall }
    | { type: 'tool_result'; toolCallId: string; toolName: string; result: ToolResult }
    | { type: 'error'; content: string }
    | { type: 'done' };

// Execute a tool and return the result
async function executeTool(toolCall: ToolCall): Promise<ToolResult> {
    const tools = getAvailableTools();
    const toolToExecute = tools.find((t: any) => t.name === toolCall.name);

    if (!toolToExecute) {
        return {
            success: false,
            error: `Tool "${toolCall.name}" not found`,
            output: null,
        };
    }

    try {
        const resultStr = await toolToExecute.invoke(toolCall.args || {});
        const parsed = JSON.parse(String(resultStr));
        return {
            success: parsed.success ?? true,
            output: parsed.output ?? null,
            error: parsed.error ?? null,
            images: parsed.images,
            code: parsed.code,
        };
    } catch (error) {
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Tool execution failed',
            output: null,
        };
    }
}

// Run the agent with proper agentic loop
export async function* runAgentLoop(
    messages: ChatMessage[]
): AsyncGenerator<StreamChunk> {
    const settings = getAgentSettings();

    if (!settings.apiKey) {
        yield { type: 'error', content: 'Please configure your Google AI Studio API key in Settings.' };
        return;
    }

    const model = createModel(settings);
    if (!model) {
        yield { type: 'error', content: 'Failed to initialize the AI model.' };
        return;
    }

    const tools = getAvailableTools();
    const modelWithTools = model.bindTools(tools);

    try {
        // Build LangChain message history
        const lcMessages: BaseMessage[] = [
            new SystemMessage(SYSTEM_PROMPT),
            ...toLangChainMessages(messages),
        ];

        // Agentic loop - continue until no more tool calls
        let iterations = 0;
        const maxIterations = 10; // Safety limit

        while (iterations < maxIterations) {
            iterations++;
            console.log(`[Agent] Iteration ${iterations}`);

            // Call the model
            const response = await modelWithTools.invoke(lcMessages);
            console.log('[Agent] Response:', response);

            // Check if response has tool calls
            const toolCalls = response.tool_calls || [];
            const hasToolCalls = toolCalls.length > 0;

            // Get text content
            const textContent = typeof response.content === 'string'
                ? response.content
                : '';

            // If there's text content, yield it
            if (textContent) {
                yield { type: 'text', content: textContent };
            }

            // If no tool calls, we're done
            if (!hasToolCalls) {
                yield { type: 'done' };
                break;
            }

            // Add the AI response to message history
            lcMessages.push(response);

            // Execute each tool call
            for (const tc of toolCalls) {
                const toolCall: ToolCall = {
                    id: tc.id || `tc_${Date.now()}`,
                    name: tc.name,
                    args: tc.args as Record<string, unknown>,
                };

                yield { type: 'tool_call', toolCall };

                // Execute the tool
                const result = await executeTool(toolCall);

                yield {
                    type: 'tool_result',
                    toolCallId: toolCall.id,
                    toolName: toolCall.name,
                    result
                };

                // Add tool result to message history
                lcMessages.push(new ToolMessage({
                    tool_call_id: toolCall.id,
                    content: JSON.stringify(result),
                }));
            }

            // Loop continues - model will see tool results
        }

        if (iterations >= maxIterations) {
            yield { type: 'error', content: 'Agent reached maximum iterations limit.' };
        }

    } catch (error) {
        console.error('[Agent] Error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        yield { type: 'error', content: `Error: ${errorMessage}` };
    }
}

// Generate a unique session ID
export function generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Generate a unique message ID
export function generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
