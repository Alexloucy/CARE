import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Box, Typography, Container, IconButton, Tooltip, Drawer, List, ListItemButton, ListItemText, Divider } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { ClockCounterClockwise, Trash, X } from '@phosphor-icons/react';
import AgentInputBar from '../../components/agent/AgentInputBar';
import ChatMessageRenderer from '../../components/agent/ChatMessageRenderer';
import MessageSkeleton from '../../components/agent/MessageSkeleton';
import { ChatMessage, AgentSession, CodeExecutionResult } from '../../types/agent';
import {
    streamAgentResponse,
    generateMessageId,
    generateSessionId,
    getAgentSettings,
    getSessions,
    saveSession,
    deleteSession,
    getCurrentSessionId,
    setCurrentSessionId,
} from '../../services/agentService';

const AgentPage: React.FC = () => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [streamingContent, setStreamingContent] = useState('');
    const [currentSessionId, setCurrentSessionIdState] = useState<string>('');
    const [sessions, setSessions] = useState<AgentSession[]>([]);
    const [historyOpen, setHistoryOpen] = useState(false);
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    // Load sessions and current session on mount
    useEffect(() => {
        const loadedSessions = getSessions();
        setSessions(loadedSessions);

        const savedSessionId = getCurrentSessionId();
        if (savedSessionId) {
            const session = loadedSessions.find(s => s.id === savedSessionId);
            if (session) {
                setCurrentSessionIdState(session.id);
                setMessages(session.messages);
                return;
            }
        }

        // Start a new session if none exists
        const newId = generateSessionId();
        setCurrentSessionIdState(newId);
        setCurrentSessionId(newId);
    }, []);

    // Save current session when messages change
    useEffect(() => {
        if (currentSessionId && messages.length > 0) {
            // Generate title from first user message
            const firstUserMsg = messages.find(m => m.role === 'user');
            const title = firstUserMsg
                ? firstUserMsg.content.slice(0, 40) + (firstUserMsg.content.length > 40 ? '...' : '')
                : 'New Chat';

            const session: AgentSession = {
                id: currentSessionId,
                title,
                messages,
                createdAt: sessions.find(s => s.id === currentSessionId)?.createdAt || new Date(),
                updatedAt: new Date(),
            };
            saveSession(session);

            // Update local sessions list
            setSessions(prev => {
                const existing = prev.findIndex(s => s.id === currentSessionId);
                if (existing >= 0) {
                    const updated = [...prev];
                    updated[existing] = session;
                    return updated;
                }
                return [session, ...prev];
            });
        }
    }, [messages, currentSessionId]);

    // Auto-scroll to bottom when new messages arrive or streaming content updates
    useEffect(() => {
        if (scrollContainerRef.current) {
            scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
    }, [messages, streamingContent]);

    const handleNewChat = useCallback(() => {
        // Save current session first if it has messages
        if (currentSessionId && messages.length > 0) {
            const firstUserMsg = messages.find(m => m.role === 'user');
            const title = firstUserMsg
                ? firstUserMsg.content.slice(0, 40) + (firstUserMsg.content.length > 40 ? '...' : '')
                : 'New Chat';
            saveSession({
                id: currentSessionId,
                title,
                messages,
                createdAt: sessions.find(s => s.id === currentSessionId)?.createdAt || new Date(),
                updatedAt: new Date(),
            });
        }

        // Create new session
        const newId = generateSessionId();
        setCurrentSessionIdState(newId);
        setCurrentSessionId(newId);
        setMessages([]);
        setStreamingContent('');
        setIsLoading(false);
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
    }, [currentSessionId, messages, sessions]);

    const handleLoadSession = useCallback((session: AgentSession) => {
        setCurrentSessionIdState(session.id);
        setCurrentSessionId(session.id);
        setMessages(session.messages);
        setStreamingContent('');
        setHistoryOpen(false);
    }, []);

    const handleDeleteSession = useCallback((sessionId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        deleteSession(sessionId);
        setSessions(prev => prev.filter(s => s.id !== sessionId));

        // If deleting current session, start a new one
        if (sessionId === currentSessionId) {
            const newId = generateSessionId();
            setCurrentSessionIdState(newId);
            setCurrentSessionId(newId);
            setMessages([]);
        }
    }, [currentSessionId]);

    const handleStopGeneration = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        // If we have streaming content, add it as a message
        if (streamingContent) {
            const assistantMessage: ChatMessage = {
                id: generateMessageId(),
                role: 'assistant',
                content: streamingContent + '\n\n*[Generation stopped]*',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, assistantMessage]);
            setStreamingContent('');
        }
        setIsLoading(false);
    }, [streamingContent]);

    const handleSendMessage = useCallback(async (content: string) => {
        // Check for API key
        const settings = getAgentSettings();
        if (!settings.apiKey) {
            const errorMessage: ChatMessage = {
                id: generateMessageId(),
                role: 'assistant',
                content: '⚠️ Please configure your Google AI Studio API key in **Settings** to use the AI agent.',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
            return;
        }

        // Add user message
        const userMessage: ChatMessage = {
            id: generateMessageId(),
            role: 'user',
            content,
            timestamp: new Date(),
        };
        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);
        setStreamingContent('');

        // Create abort controller for this request
        abortControllerRef.current = new AbortController();

        try {
            const allMessages = [...messages, userMessage];
            let accumulatedContent = '';
            let latestCodeExecution: CodeExecutionResult | null = null;

            for await (const chunk of streamAgentResponse(allMessages)) {
                // Check if aborted
                if (abortControllerRef.current?.signal.aborted) {
                    break;
                }

                if (chunk.type === 'error') {
                    const errorMessage: ChatMessage = {
                        id: generateMessageId(),
                        role: 'assistant',
                        content: `❌ ${chunk.content}`,
                        timestamp: new Date(),
                    };
                    setMessages(prev => [...prev, errorMessage]);
                    setStreamingContent('');
                    break;
                }

                if (chunk.type === 'tool') {
                    // Show tool call as a separate message
                    const toolMessage: ChatMessage = {
                        id: generateMessageId(),
                        role: 'tool',
                        content: chunk.content,
                        timestamp: new Date(),
                    };
                    setMessages(prev => [...prev, toolMessage]);
                }

                if (chunk.type === 'tool_result' && chunk.codeExecution) {
                    // Store code execution result locally to attach to the final message
                    latestCodeExecution = chunk.codeExecution;
                }

                if (chunk.type === 'text_delta') {
                    // Accumulate streaming content
                    accumulatedContent += chunk.content;
                    setStreamingContent(accumulatedContent);
                }
            }

            // After streaming completes, add the full message with code execution if present
            if (accumulatedContent && !abortControllerRef.current?.signal.aborted) {
                const assistantMessage: ChatMessage = {
                    id: generateMessageId(),
                    role: 'assistant',
                    content: accumulatedContent,
                    timestamp: new Date(),
                    codeExecution: latestCodeExecution || undefined,
                };
                setMessages(prev => [...prev, assistantMessage]);
                setStreamingContent('');
            }
        } catch (error) {
            if ((error as Error).name !== 'AbortError') {
                const errorMessage: ChatMessage = {
                    id: generateMessageId(),
                    role: 'assistant',
                    content: `❌ An error occurred: ${(error as Error).message}`,
                    timestamp: new Date(),
                };
                setMessages(prev => [...prev, errorMessage]);
            }
            setStreamingContent('');
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    }, [messages]);

    const formatSessionDate = (date: Date) => {
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));

        if (days === 0) return 'Today';
        if (days === 1) return 'Yesterday';
        if (days < 7) return `${days} days ago`;
        return date.toLocaleDateString();
    };

    return (
        <Box
            sx={{
                position: 'relative',
                height: '100vh',
                overflow: 'hidden',
            }}
        >
            {/* History toggle button */}
            <Tooltip title="Chat History">
                <IconButton
                    onClick={() => setHistoryOpen(true)}
                    sx={{
                        position: 'absolute',
                        top: 76,
                        left: 16,
                        zIndex: 10,
                        bgcolor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.04)',
                        backdropFilter: 'blur(12px)',
                        '&:hover': {
                            bgcolor: isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.08)',
                        },
                    }}
                >
                    <ClockCounterClockwise size={20} />
                </IconButton>
            </Tooltip>

            {/* History Drawer */}
            <Drawer
                anchor="left"
                open={historyOpen}
                onClose={() => setHistoryOpen(false)}
                PaperProps={{
                    sx: {
                        width: 300,
                        bgcolor: isDark ? 'rgba(30,30,36,0.95)' : 'rgba(255,255,255,0.95)',
                        backdropFilter: 'blur(20px)',
                    },
                }}
            >
                <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Typography variant="h6" fontWeight={600}>Chat History</Typography>
                    <IconButton size="small" onClick={() => setHistoryOpen(false)}>
                        <X size={18} />
                    </IconButton>
                </Box>
                <Divider />
                <List sx={{ flex: 1, overflow: 'auto' }}>
                    {sessions.length === 0 ? (
                        <Box sx={{ p: 3, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary">
                                No chat history yet
                            </Typography>
                        </Box>
                    ) : (
                        sessions.map(session => (
                            <ListItemButton
                                key={session.id}
                                selected={session.id === currentSessionId}
                                onClick={() => handleLoadSession(session)}
                                sx={{
                                    borderRadius: 1,
                                    mx: 1,
                                    mb: 0.5,
                                }}
                            >
                                <ListItemText
                                    primary={session.title}
                                    secondary={formatSessionDate(session.updatedAt)}
                                    primaryTypographyProps={{
                                        noWrap: true,
                                        fontSize: '0.875rem',
                                    }}
                                    secondaryTypographyProps={{
                                        fontSize: '0.75rem',
                                    }}
                                />
                                <IconButton
                                    size="small"
                                    onClick={(e: React.MouseEvent) => handleDeleteSession(session.id, e)}
                                    sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                                >
                                    <Trash size={16} />
                                </IconButton>
                            </ListItemButton>
                        ))
                    )}
                </List>
            </Drawer>

            {/* Messages Container - takes full page with padding for input */}
            <Box
                ref={scrollContainerRef}
                sx={{
                    position: 'absolute',
                    top: 68, // navbar height
                    left: 0,
                    right: 0,
                    bottom: 0,
                    overflowY: 'auto',
                    overflowX: 'hidden',
                    pb: '100px', // Space for input bar
                }}
            >
                <Container maxWidth="md" sx={{ py: 2 }}>
                    {messages.length === 0 && !streamingContent ? (
                        <Box
                            sx={{
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                minHeight: '40vh',
                                textAlign: 'center',
                                gap: 2,
                            }}
                        >
                            <Typography
                                variant="h4"
                                sx={{
                                    fontWeight: 600,
                                    background: isDark
                                        ? 'linear-gradient(135deg, #fff 0%, #888 100%)'
                                        : 'linear-gradient(135deg, #333 0%, #666 100%)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                }}
                            >
                                How can I help?
                            </Typography>
                            <Typography
                                variant="body1"
                                sx={{ color: theme.palette.text.secondary, maxWidth: 400 }}
                            >
                                Ask me anything about wildlife conservation, image analysis, or try asking me to reveal a secret!
                            </Typography>
                        </Box>
                    ) : (
                        <>
                            {messages.map((message) => (
                                <ChatMessageRenderer key={message.id} message={message} />
                            ))}
                            {/* Show streaming content as a temporary message */}
                            {streamingContent && (
                                <ChatMessageRenderer
                                    message={{
                                        id: 'streaming',
                                        role: 'assistant',
                                        content: streamingContent,
                                        timestamp: new Date(),
                                        isStreaming: true,
                                    }}
                                />
                            )}
                            {isLoading && !streamingContent && messages[messages.length - 1]?.role === 'user' && (
                                <MessageSkeleton />
                            )}
                        </>
                    )}
                </Container>
            </Box>

            {/* Input Bar - absolute positioned at bottom */}
            <Box
                sx={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    zIndex: 10,
                }}
            >
                <AgentInputBar
                    onSendMessage={handleSendMessage}
                    onNewChat={handleNewChat}
                    isLoading={isLoading}
                    onStopGeneration={handleStopGeneration}
                />
            </Box>
        </Box>
    );
};

export default AgentPage;
