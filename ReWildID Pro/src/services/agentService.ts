// LangChain Agent Service for Google AI Studio (Gemini)
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { ChatMessage, AgentSettings, DEFAULT_AGENT_SETTINGS, ToolCall, ToolResult, AgentSession, ConfirmationRequest } from '../types/agent';

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
        description: 'Execute Python code locally for READ-ONLY operations: calculations, data analysis, and generating visualizations. For database WRITES (updating metadata), use requestMetadataUpdate instead.',
        schema: z.object({
            code: z.string().describe('The Python code to execute. Include necessary imports like "import matplotlib.pyplot as plt". Use plt.show() to display charts.'),
        }),
    }
);

// Tool for requesting metadata updates (requires user confirmation)
const requestMetadataUpdateTool = tool(
    async ({
        description,
        filter_sql,
        update_code
    }: {
        description: string;
        filter_sql: string;
        update_code: string;
    }) => {
        console.log('[MetadataUpdate] Requesting confirmation for:', description);

        try {
            // First, query to see how many rows will be affected
            const previewCode = `
import sqlite3
import json

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT id, original_path, metadata FROM images WHERE ${filter_sql.replace(/"/g, '\\"')}")
rows = cursor.fetchall()
conn.close()

result = {
    "count": len(rows),
    "preview": [row[1] for row in rows[:5]],  # First 5 paths
    "ids": [row[0] for row in rows]
}
print(json.dumps(result))
`;
            const previewResult = await (window as any).api.executePythonCode(previewCode);

            if (!previewResult.success) {
                return JSON.stringify({
                    success: false,
                    error: previewResult.error || 'Failed to query affected rows',
                    output: null,
                    requiresConfirmation: false,
                });
            }

            const parsed = JSON.parse(previewResult.output.trim());

            // Return a special response that will trigger confirmation
            return JSON.stringify({
                success: true,
                requiresConfirmation: true,
                confirmationData: {
                    id: `confirm_${Date.now()}`,
                    action: 'update_metadata',
                    description: description,
                    affectedCount: parsed.count,
                    preview: parsed.preview,
                    pendingCode: update_code,
                    filterSql: filter_sql,  // Store the filter for backup
                    status: 'pending',
                } as ConfirmationRequest,
                output: `Found ${parsed.count} images matching the filter. Waiting for user confirmation.`,
                error: null,
            });
        } catch (error) {
            console.error('[MetadataUpdate] Exception:', error);
            return JSON.stringify({
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                output: null,
                requiresConfirmation: false,
            });
        }
    },
    {
        name: 'requestMetadataUpdate',
        description: 'Request to update image metadata in the database. This will show a confirmation dialog to the user before executing. Use this for any database WRITE operations.',
        schema: z.object({
            description: z.string().describe('Human-readable description of what will be updated, e.g., "Tag 45 forest images with location=ForestA"'),
            filter_sql: z.string().describe('SQL WHERE clause to filter images, e.g., "original_path LIKE \'%forest%\'"'),
            update_code: z.string().describe('Python code that will be executed after user confirms. This code should: 1) backup the table first, 2) then perform the update. Example: window.api.backupTable("images") then UPDATE images SET metadata...'),
        }),
    }
);

// Get the tools list
function getAvailableTools() {
    return [revealSecretTool, runPythonCodeTool as any, requestMetadataUpdateTool as any];
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
You help users with wildlife conservation tasks, image analysis, data queries, and general questions.

Available tools:
- revealSecret: Reveals a secret message when users ask for it
- runPythonCode: Execute Python code to perform calculations, data analysis, database queries, or generate charts/visualizations with matplotlib.

## Database Access
IMPORTANT: The variable \`DB_PATH\` is ALREADY DEFINED and contains the path to the SQLite database. Do NOT redefine it. Just use it directly:

\`\`\`python
import sqlite3
import pandas as pd

# DB_PATH is already defined - do NOT set it yourself!
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM images LIMIT 10", conn)
print(df)
conn.close()
\`\`\`

### Database Schema

**groups** - Image import batches
- id INTEGER PRIMARY KEY
- name TEXT (folder name)
- created_at INTEGER (timestamp ms)
- updated_at INTEGER

**images** - Individual photos
- id INTEGER PRIMARY KEY
- group_id INTEGER (FK to groups)
- original_path TEXT (absolute file path)
- preview_path TEXT (thumbnail path)
- date_added INTEGER (timestamp ms)
- metadata TEXT (JSON of EXIF data)

**detection_batches** - Classification runs
- id INTEGER PRIMARY KEY
- name TEXT
- created_at INTEGER
- updated_at INTEGER

**detections** - Animal detections in images
- id INTEGER PRIMARY KEY
- batch_id INTEGER (FK to detection_batches)
- image_id INTEGER (FK to images)
- label TEXT (species name like "deer", "boar", "bird")
- confidence REAL (0-1, species classification confidence)
- detection_confidence REAL (0-1, detection box confidence)
- x1, y1, x2, y2 REAL (bounding box coordinates 0-1)
- source TEXT
- created_at INTEGER

**reid_runs** - Re-identification sessions
- id INTEGER PRIMARY KEY
- name TEXT
- species TEXT (which species was re-identified)
- created_at INTEGER

**reid_individuals** - Individual animals identified
- id INTEGER PRIMARY KEY
- run_id INTEGER (FK to reid_runs)
- name TEXT (original ID like "ID-0")
- display_name TEXT (user-facing name like "ID-1")
- color TEXT (hex color for UI)
- created_at INTEGER

**reid_members** - Links detections to individuals
- id INTEGER PRIMARY KEY
- individual_id INTEGER (FK to reid_individuals)
- detection_id INTEGER (FK to detections)

### Common Queries
- Count images: \`SELECT COUNT(*) FROM images\`
- Species distribution: \`SELECT label, COUNT(*) FROM detections GROUP BY label\`
- Images per day: \`SELECT date(created_at/1000, 'unixepoch') as day, COUNT(*) FROM groups GROUP BY day\`
- Individual sighting counts: \`SELECT display_name, COUNT(*) FROM reid_individuals ri JOIN reid_members rm ON ri.id = rm.individual_id GROUP BY ri.id\`

## Updating Metadata
To update image metadata (tagging, adding location, etc.), use the \`requestMetadataUpdate\` tool. This requires user confirmation before executing.

Example workflow:
1. User: "Tag all forest images with location=ForestA"
2. You call requestMetadataUpdate with filter_sql and update_code
3. User sees confirmation dialog with preview of affected images
4. After confirmation, the update executes with automatic backup

The update_code should include backup first:
\`\`\`python
import sqlite3
import json

# Backup happens automatically before confirmation

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get existing metadata and merge
cursor.execute("SELECT id, metadata FROM images WHERE original_path LIKE '%forest%'")
for row in cursor.fetchall():
    existing = json.loads(row[1]) if row[1] else {}
    existing['location'] = 'ForestA'
    cursor.execute("UPDATE images SET metadata = ? WHERE id = ?", (json.dumps(existing), row[0]))

conn.commit()
conn.close()
print("Updated successfully")
\`\`\`

## Visualization Guidelines
1. Always use matplotlib.pyplot
2. Use plt.show() to display charts - images are automatically captured
3. For nice charts, use seaborn with a clean style
4. Add proper titles, labels, and legends
5. Use appropriate chart types (bar for categories, line for time series, pie for proportions)

Be friendly, concise, and helpful. Proactively use database queries when users ask about their wildlife data.`;

// Stream chunk types
export type StreamChunk =
    | { type: 'text'; content: string }
    | { type: 'tool_call'; toolCall: ToolCall }
    | { type: 'tool_result'; toolCallId: string; toolName: string; result: ToolResult }
    | { type: 'confirmation_request'; request: import('../types/agent').ConfirmationRequest }
    | { type: 'error'; content: string }
    | { type: 'done' };

// Execute a tool and return the result (with optional confirmation data)
interface ToolExecutionResult {
    result: ToolResult;
    confirmationRequest?: ConfirmationRequest;
}

async function executeTool(toolCall: ToolCall): Promise<ToolExecutionResult> {
    const tools = getAvailableTools();
    const toolToExecute = tools.find((t: any) => t.name === toolCall.name);

    if (!toolToExecute) {
        return {
            result: {
                success: false,
                error: `Tool "${toolCall.name}" not found`,
                output: null,
            }
        };
    }

    try {
        const resultStr = await toolToExecute.invoke(toolCall.args || {});
        const parsed = JSON.parse(String(resultStr));

        const result: ToolResult = {
            success: parsed.success ?? true,
            output: parsed.output ?? null,
            error: parsed.error ?? null,
            images: parsed.images,
            code: parsed.code,
        };

        // Check if this is a confirmation request
        if (parsed.requiresConfirmation && parsed.confirmationData) {
            return {
                result,
                confirmationRequest: parsed.confirmationData as ConfirmationRequest,
            };
        }

        return { result };
    } catch (error) {
        return {
            result: {
                success: false,
                error: error instanceof Error ? error.message : 'Tool execution failed',
                output: null,
            }
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

            // Get text content - handle both string and array content
            let textContent = '';
            if (typeof response.content === 'string') {
                textContent = response.content;
            } else if (Array.isArray(response.content)) {
                // Gemini sometimes returns content as array of parts
                textContent = response.content
                    .filter((part: any) => part && typeof part.text === 'string')
                    .map((part: any) => part.text)
                    .join('');
            }

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
                const { result, confirmationRequest } = await executeTool(toolCall);

                // If this is a confirmation request, yield it and stop the loop
                if (confirmationRequest) {
                    yield { type: 'confirmation_request', request: confirmationRequest };
                    yield { type: 'done' };
                    return; // Exit the generator entirely - user must respond
                }

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
        // Log full error to console for debugging
        console.error('[Agent] Error:', error);
        if (error instanceof Error) {
            console.error('[Agent] Error details:', error.message, error.stack);
        }
        // Show user-friendly message
        yield { type: 'error', content: 'The agent ran into an issue. Please try again.' };
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

// Execute a confirmed metadata update (with backup)
export async function executeConfirmedUpdate(
    confirmationRequest: ConfirmationRequest
): Promise<ToolResult> {
    console.log('[ConfirmedUpdate] Executing:', confirmationRequest.description);

    try {
        // First, backup ONLY the affected rows using the filter
        console.log('[ConfirmedUpdate] Creating backup for filter:', confirmationRequest.filterSql);
        const backupResult = await (window as any).api.backupTable(
            'images',
            confirmationRequest.filterSql  // Only backup affected rows
        );

        if (!backupResult.success) {
            return {
                success: false,
                error: `Backup failed: ${backupResult.error}`,
                output: null,
            };
        }
        console.log('[ConfirmedUpdate] Backup created:', backupResult.backupPath);

        // Now execute the update code
        const result = await (window as any).api.executePythonCode(confirmationRequest.pendingCode);

        return {
            success: result.success,
            output: result.success
                ? `✅ Updated ${confirmationRequest.affectedCount} images successfully. Backup saved.`
                : null,
            error: result.error,
            code: confirmationRequest.pendingCode,
            backupPath: backupResult.backupPath,  // Return backup path for reverting
        };
    } catch (error) {
        console.error('[ConfirmedUpdate] Exception:', error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
            output: null,
        };
    }
}

// Revert a previously applied update using backup
export async function revertUpdate(backupPath: string): Promise<ToolResult> {
    console.log('[RevertUpdate] Reverting from:', backupPath);

    try {
        const result = await (window as any).api.restoreBackup(backupPath);

        if (!result.success) {
            return {
                success: false,
                error: result.error || 'Failed to restore backup',
                output: null,
            };
        }

        return {
            success: true,
            output: `↩️ Reverted ${result.rowCount} rows from backup.`,
            error: null,
        };
    } catch (error) {
        console.error('[RevertUpdate] Exception:', error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
            output: null,
        };
    }
}
