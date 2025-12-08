// LangChain Agent Service for Google AI Studio (Gemini)
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { ChatMessage, AgentSettings, DEFAULT_AGENT_SETTINGS } from '../types/agent';
import { Sandbox } from '@e2b/code-interpreter';

// Define the secret reveal tool
const revealSecretTool = tool(
    async () => {
        return 'asoidfjaiosdfj';
    },
    {
        name: 'revealSecret',
        description: 'Reveals a secret message when the user asks for it',
        schema: z.object({}),
    }
);

// Define the Python code execution tool
const runPythonCodeTool = tool(
    async ({ code }: { code: string }) => {
        const settings = getAgentSettings();
        console.log('[E2B] Starting code execution, code length:', code.length);

        if (!settings.e2bApiKey) {
            console.log('[E2B] No API key configured');
            return JSON.stringify({
                success: false,
                error: 'E2B API key not configured. Please add it in Settings.',
                output: null,
                images: [],
            });
        }

        try {
            console.log('[E2B] Creating sandbox...');
            // Create sandbox with API key
            const sandbox = await Sandbox.create({
                apiKey: settings.e2bApiKey,
            });
            console.log('[E2B] Sandbox created');

            try {
                // Execute the code
                console.log('[E2B] Running code...');
                const execution = await sandbox.runCode(code);
                console.log('[E2B] Execution complete. Results count:', execution.results.length);
                console.log('[E2B] Stdout lines:', execution.logs.stdout.length);
                console.log('[E2B] Stderr lines:', execution.logs.stderr.length);
                console.log('[E2B] Error:', execution.error ? `${execution.error.name}: ${execution.error.value}` : 'none');

                // Collect results - properly format error using name/value/traceback
                const errorMessage = execution.error
                    ? `${execution.error.name}: ${execution.error.value}\n${execution.error.traceback || ''}`
                    : null;

                const result = {
                    success: !execution.error,
                    error: errorMessage,
                    output: [...execution.logs.stdout, ...execution.logs.stderr].join('\n'),
                    images: [] as string[],
                };

                // Check for image results (matplotlib, etc.)
                for (const output of execution.results) {
                    console.log('[E2B] Result formats:', output.formats ? output.formats() : 'N/A');
                    if (output.png) {
                        console.log('[E2B] Found PNG image, length:', output.png.length);
                        result.images.push(`data:image/png;base64,${output.png}`);
                    }
                    if (output.jpeg) {
                        console.log('[E2B] Found JPEG image');
                        result.images.push(`data:image/jpeg;base64,${output.jpeg}`);
                    }
                }

                console.log('[E2B] Final result - success:', result.success, 'images:', result.images.length);
                return JSON.stringify(result);
            } finally {
                // Always close the sandbox
                console.log('[E2B] Killing sandbox...');
                await sandbox.kill();
            }
        } catch (error) {
            console.error('[E2B] Exception:', error);
            return JSON.stringify({
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                output: null,
                images: [],
            });
        }
    },
    {
        name: 'runPythonCode',
        description: 'Execute Python code in a secure sandbox. Use this for calculations, data processing, and generating visualizations with matplotlib/seaborn. Code output and any generated images will be returned.',
        schema: z.object({
            code: z.string().describe('The Python code to execute. Include necessary imports. For charts, use matplotlib.pyplot and they will be returned as images.'),
        }),
    }
);

// Get the tools list based on available API keys
function getAvailableTools() {
    const settings = getAgentSettings();
    const availableTools = [revealSecretTool];

    if (settings.e2bApiKey) {
        availableTools.push(runPythonCodeTool as any);
    }

    return availableTools;
}

// Storage keys
const SETTINGS_KEY = 'agent_settings';
const SESSIONS_KEY = 'agent_sessions';
const CURRENT_SESSION_KEY = 'agent_current_session';

// Session management
export function getSessions(): import('../types/agent').AgentSession[] {
    try {
        const stored = localStorage.getItem(SESSIONS_KEY);
        if (stored) {
            const sessions = JSON.parse(stored);
            // Convert date strings back to Date objects
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

export function saveSession(session: import('../types/agent').AgentSession): void {
    const sessions = getSessions();
    const existingIdx = sessions.findIndex(s => s.id === session.id);
    if (existingIdx >= 0) {
        sessions[existingIdx] = session;
    } else {
        sessions.unshift(session); // Add to beginning
    }
    // Keep only last 50 sessions
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
        model: settings.model || 'gemini-flash-latest',
        maxOutputTokens: 8192,
        temperature: 0.7,
        streaming: true, // Enable streaming
    });
}

// Convert our messages to LangChain format
function toLangChainMessages(messages: ChatMessage[]): BaseMessage[] {
    return messages
        .filter((m) => m.role !== 'tool') // Tool messages are handled separately
        .map((m) => {
            if (m.role === 'user') {
                return new HumanMessage(m.content);
            }
            return new AIMessage(m.content);
        });
}

// System prompt for the agent
const SYSTEM_PROMPT = `You are a helpful AI assistant for RewildID Pro, a wildlife re-identification application. 
You help users with wildlife conservation tasks, image analysis, and general questions.

Available tools:
- revealSecret: Reveals a secret message when users ask for it
- runPythonCode: Execute Python code to perform calculations, data analysis, or generate charts/visualizations with matplotlib. Use this tool when users want to see graphs, charts, plots, or need computational tasks done.

When generating visualizations:
1. Always use matplotlib.pyplot
2. Use plt.savefig() is NOT needed - images are automatically captured
3. For nice charts, consider using seaborn
4. Add proper titles, labels, and legends

Be friendly, concise, and helpful. When users ask for data visualization or charts, proactively use the runPythonCode tool.`;

// Stream a response from the agent with real token streaming
export async function* streamAgentResponse(
    messages: ChatMessage[],
    onToolCall?: (toolName: string, args: Record<string, unknown>) => void
): AsyncGenerator<{
    type: 'text' | 'text_delta' | 'tool' | 'tool_result' | 'error';
    content: string;
    toolName?: string;
    toolArgs?: Record<string, unknown>;
    codeExecution?: import('../types/agent').CodeExecutionResult;
}> {
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
        const langChainMessages: BaseMessage[] = [
            new SystemMessage(SYSTEM_PROMPT),
            ...toLangChainMessages(messages),
        ];

        // Use streaming
        const stream = await modelWithTools.stream(langChainMessages);

        let fullContent = '';
        let toolCalls: any[] = [];

        for await (const chunk of stream) {
            // Handle text content streaming
            if (chunk.content) {
                const textContent = typeof chunk.content === 'string'
                    ? chunk.content
                    : '';
                if (textContent) {
                    fullContent += textContent;
                    yield { type: 'text_delta', content: textContent };
                }
            }

            // Collect tool calls
            if (chunk.tool_calls && chunk.tool_calls.length > 0) {
                toolCalls = chunk.tool_calls;
            }
        }

        // Handle tool calls after streaming completes
        if (toolCalls.length > 0) {
            for (const toolCall of toolCalls) {
                yield {
                    type: 'tool',
                    content: `Calling tool: ${toolCall.name}`,
                    toolName: toolCall.name,
                    toolArgs: toolCall.args as Record<string, unknown>,
                };

                if (onToolCall) {
                    onToolCall(toolCall.name, toolCall.args as Record<string, unknown>);
                }

                // Execute the tool
                const toolToExecute = tools.find((t: any) => t.name === toolCall.name);
                if (toolToExecute) {
                    const toolResult = await toolToExecute.invoke(toolCall.args || {});

                    // If this is a code execution tool, yield the result with code and images
                    if (toolCall.name === 'runPythonCode') {
                        try {
                            const parsedResult = JSON.parse(String(toolResult));
                            yield {
                                type: 'tool_result' as any,
                                content: 'Code execution complete',
                                toolName: toolCall.name,
                                codeExecution: {
                                    success: parsedResult.success,
                                    error: parsedResult.error,
                                    output: parsedResult.output,
                                    images: parsedResult.images || [],
                                    code: (toolCall.args as any)?.code || '',
                                },
                            };
                        } catch {
                            // If parsing fails, just continue
                        }
                    }

                    // Add the tool result to messages and get final response with streaming
                    const messagesWithTool = [
                        ...langChainMessages,
                        new AIMessage({ content: fullContent, tool_calls: toolCalls }),
                        new ToolMessage({
                            tool_call_id: toolCall.id || 'unknown',
                            content: String(toolResult),
                        }),
                    ];

                    // Stream the final response after tool execution
                    const finalStream = await model.stream(messagesWithTool);
                    for await (const finalChunk of finalStream) {
                        if (finalChunk.content) {
                            const textContent = typeof finalChunk.content === 'string'
                                ? finalChunk.content
                                : '';
                            if (textContent) {
                                yield { type: 'text_delta', content: textContent };
                            }
                        }
                    }
                }
            }
        }
    } catch (error) {
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
