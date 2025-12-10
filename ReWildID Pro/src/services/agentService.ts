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

// Tool for requesting classification (detection) job
const requestClassificationTool = tool(
    async ({ image_ids, description }: { image_ids: number[]; description: string }) => {
        console.log('[Classification] Requesting confirmation for:', description);

        try {
            // Get image info for preview - returns { ok, images } or { ok: false, error }
            const response = await (window as any).api.getImagesByIds(image_ids);
            if (!response.ok || !response.images || response.images.length === 0) {
                return JSON.stringify({
                    success: false,
                    error: response.error || 'No valid images found with the provided IDs',
                    output: null,
                    requiresConfirmation: false,
                });
            }
            const images = response.images;

            // Return confirmation request
            return JSON.stringify({
                success: true,
                requiresConfirmation: true,
                confirmationData: {
                    id: `confirm_classify_${Date.now()}`,
                    action: 'run_classification',
                    description: description,
                    affectedCount: images.length,
                    preview: images.slice(0, 5).map((img: any) => img.original_path),
                    imageIds: image_ids,
                    status: 'pending',
                } as ConfirmationRequest,
                output: `Ready to classify ${images.length} images. Waiting for user confirmation.`,
                error: null,
            });
        } catch (error) {
            console.error('[Classification] Exception:', error);
            return JSON.stringify({
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                output: null,
                requiresConfirmation: false,
            });
        }
    },
    {
        name: 'requestClassification',
        description: 'Request to run species classification (detection) on images. This will show a confirmation dialog to the user before starting the job.',
        schema: z.object({
            image_ids: z.array(z.number()).describe('Array of image IDs to classify'),
            description: z.string().describe('Human-readable description, e.g., "Run classification on 50 forest images"'),
        }),
    }
);

// Tool for requesting ReID job
const requestReIDTool = tool(
    async ({ image_ids, species, description }: { image_ids: number[]; species: string; description: string }) => {
        console.log('[ReID] Requesting confirmation for:', description);

        try {
            // Get image info for preview
            const response = await (window as any).api.getImagesByIds(image_ids);
            if (!response.ok || !response.images || response.images.length === 0) {
                return JSON.stringify({
                    success: false,
                    error: response.error || 'No valid images found with the provided IDs',
                    output: null,
                    requiresConfirmation: false,
                });
            }
            const images = response.images;

            // Return confirmation request
            return JSON.stringify({
                success: true,
                requiresConfirmation: true,
                confirmationData: {
                    id: `confirm_reid_${Date.now()}`,
                    action: 'run_reid',
                    description: description,
                    affectedCount: images.length,
                    preview: images.slice(0, 5).map((img: any) => img.original_path),
                    imageIds: image_ids,
                    species: species,
                    status: 'pending',
                } as ConfirmationRequest,
                output: `Ready to run ReID for "${species}" on ${images.length} images. If images need classification first, it will run automatically. Waiting for user confirmation.`,
                error: null,
            });
        } catch (error) {
            console.error('[ReID] Exception:', error);
            return JSON.stringify({
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                output: null,
                requiresConfirmation: false,
            });
        }
    },
    {
        name: 'requestReID',
        description: 'Request to run re-identification (ReID) on images to identify individual animals. If images need classification first, it will run automatically. This will show a confirmation dialog to the user before starting.',
        schema: z.object({
            image_ids: z.array(z.number()).describe('Array of image IDs for ReID'),
            species: z.string().describe('Species to identify, e.g., "deer", "boar", "wild pig"'),
            description: z.string().describe('Human-readable description, e.g., "Identify individual deer in 30 images"'),
        }),
    }
);

// Image Generation Tool (uses Gemini's native image generation - separate from LangChain)
const generateImageTool = tool(
    async ({ prompt, aspectRatio = '1:1' }: { prompt: string; aspectRatio?: string }) => {
        console.log('[ImageGen] Generating image with prompt:', prompt.substring(0, 100));

        try {
            const settings = getAgentSettings();
            if (!settings.apiKey) {
                return JSON.stringify({
                    success: false,
                    error: 'API key not configured',
                    output: null,
                    images: [],
                });
            }

            const model = settings.imageGenerationModel || 'gemini-2.5-flash-image';
            const resolution = settings.imageResolution || '1K';
            const isPro = model.includes('pro');

            // Use fetch to call Gemini API directly for image generation
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${settings.apiKey}`;

            const requestBody: any = {
                contents: [{ parts: [{ text: prompt }] }],
                generationConfig: {
                    responseModalities: ['IMAGE', 'TEXT'],
                },
            };

            // Add image config for aspect ratio and resolution
            if (isPro) {
                requestBody.generationConfig.imageConfig = {
                    aspectRatio: aspectRatio,
                    imageSize: resolution,
                };
            }

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('[ImageGen] API error:', errorText);
                return JSON.stringify({
                    success: false,
                    error: `API error: ${response.status} - ${errorText}`,
                    output: null,
                    images: [],
                });
            }

            const data = await response.json();
            const parts = data.candidates?.[0]?.content?.parts || [];
            const images: string[] = [];
            let textOutput = '';

            for (const part of parts) {
                if (part.inlineData) {
                    const base64 = part.inlineData.data;
                    const mimeType = part.inlineData.mimeType || 'image/png';
                    images.push(`data:${mimeType};base64,${base64}`);
                } else if (part.text) {
                    textOutput += part.text;
                }
            }

            if (images.length === 0) {
                return JSON.stringify({
                    success: false,
                    error: 'No image generated. The model may have declined due to content policy.',
                    output: textOutput || null,
                    images: [],
                });
            }

            console.log('[ImageGen] Generated', images.length, 'image(s)');
            return JSON.stringify({
                success: true,
                output: textOutput || `Generated ${images.length} image(s)`,
                error: null,
                images: images,
            });
        } catch (error) {
            console.error('[ImageGen] Exception:', error);
            return JSON.stringify({
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                output: null,
                images: [],
            });
        }
    },
    {
        name: 'generateImage',
        description: 'Generate an image from a text description using Gemini\'s native image generation. Returns the generated image.',
        schema: z.object({
            prompt: z.string().describe('Detailed description of the image to generate. Be specific about style, lighting, composition, etc.'),
            aspectRatio: z.string().optional().describe('Aspect ratio: "1:1", "16:9", "9:16", "4:3", "3:4" etc. Default "1:1"'),
        }),
    }
);

// Image Editing Tool (uses attached images from the conversation)
// We need to store current user images in a module-level context so the tool can access them
let currentUserImages: string[] = [];

// Function to set current user images (called before agent loop)
export function setCurrentUserImages(images: string[]) {
    currentUserImages = images;
    console.log('[ImageEdit] Set current user images:', images.length);
}

const editImageTool = tool(
    async ({ prompt, imageIndex = 0 }: { prompt: string; imageIndex?: number }) => {
        console.log('[ImageEdit] Editing image with prompt:', prompt.substring(0, 100), 'imageIndex:', imageIndex);

        try {
            const settings = getAgentSettings();
            if (!settings.apiKey) {
                return JSON.stringify({
                    success: false,
                    error: 'API key not configured',
                    output: null,
                    images: [],
                });
            }

            // Get image from current context
            if (currentUserImages.length === 0) {
                return JSON.stringify({
                    success: false,
                    error: 'No image attached to the message. Please ask the user to attach an image first.',
                    output: null,
                    images: [],
                });
            }

            if (imageIndex < 0 || imageIndex >= currentUserImages.length) {
                return JSON.stringify({
                    success: false,
                    error: `Invalid image index. User has ${currentUserImages.length} image(s) attached (indices 0-${currentUserImages.length - 1}).`,
                    output: null,
                    images: [],
                });
            }

            const imageDataUrl = currentUserImages[imageIndex];
            console.log('[ImageEdit] Using image at index', imageIndex, 'length:', imageDataUrl.length);

            const model = settings.imageGenerationModel || 'gemini-2.5-flash-image';
            const resolution = settings.imageResolution || '1K';
            const isPro = model.includes('pro');

            // Extract base64 and mime type from data URL
            const match = imageDataUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (!match) {
                return JSON.stringify({
                    success: false,
                    error: 'Invalid image format in attached image',
                    output: null,
                    images: [],
                });
            }
            const [, mimeType, base64Data] = match;

            // Use fetch to call Gemini API directly for image editing
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${settings.apiKey}`;

            const requestBody: any = {
                contents: [{
                    parts: [
                        { text: prompt },
                        { inlineData: { mimeType, data: base64Data } },
                    ]
                }],
                generationConfig: {
                    responseModalities: ['IMAGE', 'TEXT'],
                },
            };

            // Add image config for resolution (Pro only)
            if (isPro) {
                requestBody.generationConfig.imageConfig = {
                    imageSize: resolution,
                };
            }

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('[ImageEdit] API error:', errorText);
                return JSON.stringify({
                    success: false,
                    error: `API error: ${response.status} - ${errorText}`,
                    output: null,
                    images: [],
                });
            }

            const data = await response.json();
            const parts = data.candidates?.[0]?.content?.parts || [];
            const images: string[] = [];
            let textOutput = '';

            for (const part of parts) {
                if (part.inlineData) {
                    const base64 = part.inlineData.data;
                    const partMime = part.inlineData.mimeType || 'image/png';
                    images.push(`data:${partMime};base64,${base64}`);
                } else if (part.text) {
                    textOutput += part.text;
                }
            }

            if (images.length === 0) {
                return JSON.stringify({
                    success: false,
                    error: 'No image generated. The model may have declined due to content policy.',
                    output: textOutput || null,
                    images: [],
                });
            }

            console.log('[ImageEdit] Generated', images.length, 'edited image(s)');
            return JSON.stringify({
                success: true,
                output: textOutput || `Edited image generated`,
                error: null,
                images: images,
            });
        } catch (error) {
            console.error('[ImageEdit] Exception:', error);
            return JSON.stringify({
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                output: null,
                images: [],
            });
        }
    },
    {
        name: 'editImage',
        description: 'Edit an image that the user has attached to their message. Use this to modify, enhance, or transform user-provided images based on their request.',
        schema: z.object({
            prompt: z.string().describe('Description of how to edit the image. Be specific about what to add, remove, or change.'),
            imageIndex: z.number().optional().describe('Index of the attached image to edit (0 for first image, 1 for second, etc.). Default is 0.'),
        }),
    }
);

// Get the tools list
function getAvailableTools() {
    return [revealSecretTool, runPythonCodeTool as any, requestMetadataUpdateTool as any, requestClassificationTool as any, requestReIDTool as any, generateImageTool as any, editImageTool as any];
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
        maxRetries: 0, // Disable retries so we can catch 429 immediately
    });
}

// Convert our messages to LangChain format
function toLangChainMessages(messages: ChatMessage[]): BaseMessage[] {
    const result: BaseMessage[] = [];

    for (const m of messages) {
        if (m.role === 'user') {
            // Handle multimodal messages with images
            if (m.images && m.images.length > 0) {
                const content: Array<{ type: string; text?: string; image_url?: { url: string } }> = [];

                // Add text content if present
                if (m.content) {
                    content.push({ type: 'text', text: m.content });
                }

                // Add images as base64 data URLs
                for (const imageDataUrl of m.images) {
                    content.push({
                        type: 'image_url',
                        image_url: { url: imageDataUrl }
                    });
                }

                result.push(new HumanMessage({ content }));
            } else {
                result.push(new HumanMessage(m.content));
            }
        } else if (m.role === 'tool') {
            // Tool result message
            if (m.toolCallId) {
                // Strip images from tool results to prevent token limit issues
                // Images are still stored in the toolResult for display, but we don't send
                // the base64 data back to the LLM
                let toolResultForLLM: any = m.toolResult || { success: true, output: m.content };
                if (toolResultForLLM.images && toolResultForLLM.images.length > 0) {
                    // Create a copy without the images array, just reference the count
                    const { images, ...rest } = toolResultForLLM;
                    toolResultForLLM = {
                        ...rest,
                        imagesGenerated: images.length, // Just tell the LLM how many images were generated
                    };
                }
                result.push(new ToolMessage({
                    tool_call_id: m.toolCallId,
                    content: JSON.stringify(toolResultForLLM),
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
- requestMetadataUpdate: Update image metadata (requires user confirmation)
- requestClassification: Run species classification/detection on images (requires user confirmation)
- requestReID: Run re-identification to identify individual animals (requires user confirmation). If images need classification first, it runs automatically.
  - IMPORTANT: Currently only "stoat" is supported for ReID (use lowercase singular, NOT "stoats"). For other species, inform the user that ReID support is coming soon.

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

**embeddings** - Cached feature vectors for classification and ReID. This is an advanced feature, only access it when the user explicitly requests it.
- id INTEGER PRIMARY KEY
- image_id INTEGER (FK to images)
- bbox_hash TEXT (hash of bounding box coordinates to identify the crop)
- embedding_type TEXT (model type, e.g., "dinov2")
- embedding BLOB (binary numpy array of the feature vector)
- created_at INTEGER
- UNIQUE INDEX on (image_id, bbox_hash, embedding_type)

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

The update_code should follow this pattern:

**CRITICAL**: When iterating over rows and updating in the same loop, you MUST call \`fetchall()\` FIRST to load all rows into memory. If you iterate directly over \`cursor.execute()\` and call \`execute()\` again inside the loop, the cursor gets reset and only the first row is processed!

WRONG (only updates first row):
\`\`\`python
for row in cursor.execute("SELECT ..."):  # Iterating directly over execute
    cursor.execute("UPDATE ...")  # This RESETS the cursor! Loop stops after 1 row
\`\`\`

CORRECT (updates all rows):
\`\`\`python
cursor.execute("SELECT ...")
rows = cursor.fetchall()  # Load ALL rows into memory first
for row in rows:
    cursor.execute("UPDATE ...")  # Safe - we're iterating over a list, not the cursor
\`\`\`

Full example:
\`\`\`python
import sqlite3
import json

# Backup happens automatically before confirmation

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# IMPORTANT: Use fetchall() to get all rows BEFORE iterating
cursor.execute("SELECT id, metadata FROM images WHERE original_path LIKE '%forest%'")
rows = cursor.fetchall()  # Load all rows into memory first!

for row in rows:
    existing = json.loads(row[1]) if row[1] else {}
    existing['location'] = 'ForestA'
    cursor.execute("UPDATE images SET metadata = ? WHERE id = ?", (json.dumps(existing), row[0]))

conn.commit()
conn.close()
print(f"Updated {len(rows)} images successfully")
\`\`\`

## Visualization Guidelines
1. Always use matplotlib.pyplot
2. Use plt.show() to display charts - images are automatically captured
3. For nice charts, use seaborn with a clean style
4. Add proper titles, labels, and legends
5. Use appropriate chart types (bar for categories, line for time series, pie for proportions)

## ADVANCED: Pipeline Technical Details
**IMPORTANT**: This section contains internal technical details. Do NOT mention this to users unless they explicitly ask about how the system works, embeddings, model architecture, or pipeline internals. For normal users, just describe features at a high level.

### Detection Phase
1. **MegaDetector**: First pass to detect potential animals in images (bounding boxes)
2. **Animal Verification**: Check if the detection actually contains an animal
3. **Species Classification**: A 24-species classifier (DINOv3 backbone) classifies each detection
4. **Embedding Storage**: The classifier's backbone produces feature vectors stored as \`embedding_type = "dinov3_raw"\` in the embeddings table

### ReID Phase
1. **Input**: Uses cropped detections from the Detection phase
2. **Model Architecture**: Novel ReID model = DINOv3 backbone + day/night detection + species-specific adapter
3. **Embedding Caching**:
   - First checks for cached \`dinov3_reid_{species}\` embeddings (e.g., \`dinov3_reid_stoat\`)
   - If found: uses cached embedding directly (fast path)
   - If not found: takes \`dinov3_raw\` from classification, passes through the adapter, then caches the result
4. **Current Support**: Only "stoat" is supported for ReID currently

### Embedding Types in Database
- \`dinov3_raw\`: Raw backbone features from the 24-species classifier
- \`dinov3_reid_stoat\`: Processed features for stoat re-identification (after adapter)
- Future: \`dinov3_reid_{species}\` for other species as support is added

Be friendly, concise, and helpful. Proactively use database queries when users ask about their wildlife data.`;

// Stream chunk types
export type StreamChunk =
    | { type: 'text'; content: string }
    | { type: 'text_delta'; content: string }  // Streaming text chunk
    | { type: 'text_done' }  // End of streaming text
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

            // Use non-streaming invoke (streaming has bugs with some Gemini models)
            let fullContent = '';
            let toolCalls: any[] = [];

            try {
                const response = await modelWithTools.invoke(lcMessages);

                // Extract content
                if (typeof response.content === 'string') {
                    fullContent = response.content;
                } else if (Array.isArray(response.content)) {
                    fullContent = response.content
                        .filter((part: any) => part && typeof part.text === 'string')
                        .map((part: any) => part.text)
                        .join('');
                }

                // Extract tool calls
                if (response.tool_calls && response.tool_calls.length > 0) {
                    toolCalls = response.tool_calls;
                }

                // Yield the text content (all at once since not streaming)
                if (fullContent) {
                    yield { type: 'text_delta', content: fullContent };
                    yield { type: 'text_done' };
                }
            } catch (invokeError) {
                console.error('[Agent] Invoke error:', invokeError);
                // Check for 429 rate limit errors
                const errorMessage = invokeError instanceof Error ? invokeError.message : String(invokeError);
                if (errorMessage.includes('429') || errorMessage.toLowerCase().includes('rate limit') || errorMessage.toLowerCase().includes('quota')) {
                    yield { type: 'error', content: 'The AI provider is at capacity. Please check your API quota or try again in a few moments.' };
                    return; // Terminate immediately
                }
                throw invokeError;
            }

            console.log('[Agent] Full response:', fullContent, 'Tool calls:', toolCalls.length);

            const hasToolCalls = toolCalls.length > 0;

            // If no tool calls, we're done
            if (!hasToolCalls) {
                yield { type: 'done' };
                break;
            }

            // Add the AI response to message history
            lcMessages.push(new AIMessage({
                content: fullContent,
                tool_calls: toolCalls.map(tc => ({
                    id: tc.id,
                    name: tc.name,
                    args: tc.args,
                })),
            }));

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

                // Stop after image generation/editing - no need to continue the loop
                if ((toolCall.name === 'generateImage' || toolCall.name === 'editImage') && result.success) {
                    console.log('[Agent] Stopping after successful image generation');
                    yield { type: 'done' };
                    return;
                }

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
    console.log('[ConfirmedUpdate] Executing:', confirmationRequest.description, 'action:', confirmationRequest.action);

    try {
        // Handle job actions (classification, reid)
        if (confirmationRequest.action === 'run_classification') {
            if (!confirmationRequest.imageIds || confirmationRequest.imageIds.length === 0) {
                return { success: false, error: 'No image IDs provided', output: null };
            }

            // Get image paths
            const images = await (window as any).api.getImagesByIds(confirmationRequest.imageIds);
            const paths = images.map((img: any) => img.original_path);

            // Queue detection job
            await (window as any).api.detect(paths, () => { }, confirmationRequest.imageIds);

            return {
                success: true,
                output: `🚀 Classification job queued for ${confirmationRequest.imageIds.length} images. Check the Jobs panel for progress.`,
                error: null,
            };
        }

        if (confirmationRequest.action === 'run_reid') {
            if (!confirmationRequest.imageIds || confirmationRequest.imageIds.length === 0) {
                return { success: false, error: 'No image IDs provided', output: null };
            }
            if (!confirmationRequest.species) {
                return { success: false, error: 'No species specified', output: null };
            }

            // Queue ReID job (smartReID handles detection if needed)
            const result = await (window as any).api.smartReID(confirmationRequest.imageIds, confirmationRequest.species);

            if (!result.ok) {
                return { success: false, error: result.error, output: null };
            }

            return {
                success: true,
                output: `🚀 ReID job queued for ${confirmationRequest.imageIds.length} images targeting "${confirmationRequest.species}". Check the Jobs panel for progress.`,
                error: null,
            };
        }

        // Handle metadata update (existing logic)
        if (!confirmationRequest.filterSql || !confirmationRequest.pendingCode) {
            return { success: false, error: 'Missing filter or code for metadata update', output: null };
        }

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
