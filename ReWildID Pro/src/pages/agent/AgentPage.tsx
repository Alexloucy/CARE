import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Box, Typography, Container, IconButton, Tooltip, Drawer, List, ListItemButton, ListItemText, Divider } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useLocation } from 'react-router-dom';
import { ClockCounterClockwise, Trash, X, Plus } from '@phosphor-icons/react';
import AgentInputBar from '../../components/agent/AgentInputBar';
import ChatMessageRenderer from '../../components/agent/ChatMessageRenderer';
import MessageSkeleton from '../../components/agent/MessageSkeleton';
import { ChatMessage, AgentSession, ToolCall } from '../../types/agent';
import {
    runAgentLoop,
    generateMessageId,
    generateSessionId,
    getAgentSettings,
    getSessions,
    saveSession,
    deleteSession,
    getCurrentSessionId,
    setCurrentSessionId,
    executeConfirmedUpdate,
    revertUpdate,
    setCurrentUserImages,
} from '../../services/agentService';

const AgentPage: React.FC = () => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [currentSessionId, setCurrentSessionIdState] = useState<string>('');
    const [sessions, setSessions] = useState<AgentSession[]>([]);
    const [historyOpen, setHistoryOpen] = useState(false);
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);
    const location = useLocation();
    const initialMessageProcessed = useRef(false);

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

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (scrollContainerRef.current) {
            scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
    }, [messages]);

    const handleNewChat = useCallback(() => {
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

        const newId = generateSessionId();
        setCurrentSessionIdState(newId);
        setCurrentSessionId(newId);
        setMessages([]);
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
        setHistoryOpen(false);
    }, []);

    const handleDeleteSession = useCallback((sessionId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        deleteSession(sessionId);
        setSessions(prev => prev.filter(s => s.id !== sessionId));

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
        setIsLoading(false);
    }, []);

    const addMessage = useCallback((msg: Partial<ChatMessage> & { role: ChatMessage['role']; content: string }) => {
        const newMsg: ChatMessage = {
            id: generateMessageId(),
            timestamp: new Date(),
            ...msg,
        };
        setMessages(prev => [...prev, newMsg]);
        return newMsg;
    }, []);

    const handleSendMessage = useCallback(async (content: string, images?: string[]) => {
        const settings = getAgentSettings();
        if (!settings.apiKey) {
            addMessage({
                role: 'assistant',
                content: '⚠️ Please configure your Google AI Studio API key in **Settings** to use the AI agent.',
            });
            return;
        }

        // Add user message (with optional images)
        const userMessage = addMessage({ role: 'user', content, images });
        setIsLoading(true);

        abortControllerRef.current = new AbortController();

        try {
            const allMessages = [...messages, userMessage];
            let pendingToolCalls: ToolCall[] = [];
            let streamingMessageId: string | null = null;
            let streamingContent = '';

            // Set current user images for the editImage tool to access
            setCurrentUserImages(images || []);

            for await (const chunk of runAgentLoop(allMessages)) {
                if (abortControllerRef.current?.signal.aborted) {
                    break;
                }

                if (chunk.type === 'error') {
                    addMessage({
                        role: 'assistant',
                        content: `❌ ${chunk.content}`,
                    });
                    break;
                }

                if (chunk.type === 'tool_call') {
                    // Collect tool calls - they'll be added to the assistant message
                    pendingToolCalls.push(chunk.toolCall);

                    // Add assistant message showing tool call
                    addMessage({
                        role: 'assistant',
                        content: '',
                        toolCalls: [...pendingToolCalls],
                    });
                }

                if (chunk.type === 'tool_result') {
                    // Add tool result as separate message
                    addMessage({
                        role: 'tool',
                        content: '',
                        toolCallId: chunk.toolCallId,
                        toolName: chunk.toolName,
                        toolResult: chunk.result,
                    });
                    // Reset pending tool calls for next round
                    pendingToolCalls = [];
                }

                if (chunk.type === 'text_delta') {
                    // Streaming text chunk
                    streamingContent += chunk.content;

                    if (!streamingMessageId) {
                        // Create new streaming message
                        const msg = addMessage({
                            role: 'assistant',
                            content: streamingContent,
                            isStreaming: true,
                        });
                        streamingMessageId = msg.id;
                    } else {
                        // Update existing message with new content
                        setMessages(prev => prev.map(msg =>
                            msg.id === streamingMessageId
                                ? { ...msg, content: streamingContent }
                                : msg
                        ));
                    }
                }

                if (chunk.type === 'text_done') {
                    // Finalize the streaming message
                    if (streamingMessageId) {
                        setMessages(prev => prev.map(msg =>
                            msg.id === streamingMessageId
                                ? { ...msg, isStreaming: false }
                                : msg
                        ));
                    }
                    streamingMessageId = null;
                    streamingContent = '';
                }

                if (chunk.type === 'text') {
                    // Non-streaming text response (fallback)
                    addMessage({
                        role: 'assistant',
                        content: chunk.content,
                    });
                }

                if (chunk.type === 'confirmation_request') {
                    // Add inline confirmation message (no dialog)
                    addMessage({
                        role: 'confirmation',
                        content: chunk.request.description,
                        confirmationRequest: chunk.request,
                    });
                }
            }
        } catch (error) {
            if ((error as Error).name !== 'AbortError') {
                addMessage({
                    role: 'assistant',
                    content: `❌ An error occurred: ${(error as Error).message}`,
                });
            }
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    }, [messages, addMessage]);

    // Handle initialMessage from navigation (e.g., from Ask AI search bar)
    useEffect(() => {
        const state = location.state as { initialMessage?: string } | null;
        if (state?.initialMessage && !initialMessageProcessed.current && currentSessionId) {
            initialMessageProcessed.current = true;
            // Start a new chat and send the message
            const newId = generateSessionId();
            setCurrentSessionIdState(newId);
            setCurrentSessionId(newId);
            setMessages([]);
            // Use setTimeout to ensure state is updated before sending
            setTimeout(() => {
                handleSendMessage(state.initialMessage!);
            }, 100);
            // Clear the state to prevent re-sending on navigation
            window.history.replaceState({}, document.title);
        }
    }, [location.state, currentSessionId, handleSendMessage]);

    const formatSessionDate = (date: Date) => {
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));

        if (days === 0) return 'Today';
        if (days === 1) return 'Yesterday';
        if (days < 7) return `${days} days ago`;
        return date.toLocaleDateString();
    };

    // Helper to update a specific message in state
    const updateMessage = useCallback((messageId: string, updates: Partial<ChatMessage>) => {
        setMessages(prev => prev.map(msg =>
            msg.id === messageId ? { ...msg, ...updates } : msg
        ));
    }, []);

    // Handle applying a confirmation inline (called from ChatMessageRenderer)
    const handleApplyConfirmation = useCallback(async (messageId: string, confirmationRequest: ChatMessage['confirmationRequest']) => {
        if (!confirmationRequest) return;

        setIsLoading(true);

        try {
            const result = await executeConfirmedUpdate(confirmationRequest);

            if (result.success && result.backupPath) {
                // Update the confirmation message to show "applied" status
                updateMessage(messageId, {
                    confirmationRequest: {
                        ...confirmationRequest,
                        status: 'applied',
                        backupPath: result.backupPath,
                    }
                });
            }

            // Add result as assistant message (not tool - avoids LangChain issues)
            addMessage({
                role: 'assistant',
                content: result.success
                    ? `Update applied successfully. ${result.output || ''}`
                    : `Update failed: ${result.error || 'Unknown error'}`,
            });
        } catch (error) {
            addMessage({
                role: 'assistant',
                content: `❌ Failed to execute update: ${(error as Error).message}`,
            });
        } finally {
            setIsLoading(false);
        }
    }, [updateMessage, addMessage]);

    // Handle reverting an applied confirmation
    const handleRevertConfirmation = useCallback(async (messageId: string, confirmationRequest: ChatMessage['confirmationRequest']) => {
        if (!confirmationRequest || !confirmationRequest.backupPath) return;

        setIsLoading(true);

        try {
            const result = await revertUpdate(confirmationRequest.backupPath);

            if (result.success) {
                // Update the confirmation message to show "reverted" status
                updateMessage(messageId, {
                    confirmationRequest: {
                        ...confirmationRequest,
                        status: 'reverted',
                    }
                });
            }

            // Add result as assistant message (not tool - avoids LangChain issues)
            addMessage({
                role: 'assistant',
                content: result.success
                    ? `Revert completed. ${result.output || ''}`
                    : `Revert failed: ${result.error || 'Unknown error'}`,
            });
        } catch (error) {
            addMessage({
                role: 'assistant',
                content: `Failed to revert: ${(error as Error).message}`,
            });
        } finally {
            setIsLoading(false);
        }
    }, [updateMessage, addMessage]);

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
                <Box sx={{ px: 2, py: 1.5 }}>
                    <IconButton
                        onClick={() => { handleNewChat(); setHistoryOpen(false); }}
                        disabled={isLoading}
                        sx={{
                            width: '100%',
                            borderRadius: 2,
                            py: 1,
                            gap: 1,
                            bgcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)',
                            '&:hover': {
                                bgcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)',
                            },
                        }}
                    >
                        <Plus size={18} weight="bold" />
                        <Typography variant="body2" fontWeight={500}>New Chat</Typography>
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

            {/* Messages Container */}
            <Box
                ref={scrollContainerRef}
                sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    overflowY: 'auto',
                    overflowX: 'hidden',
                    pb: '100px',
                }}
            >
                <Container maxWidth="md" sx={{ py: 2 }}>
                    {/* Spacer to push content below navbar */}
                    <Box sx={{ height: 68 }} />
                    {messages.length === 0 ? (
                        <Box
                            sx={{
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                minHeight: '71vh',
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
                                Ask me anything about wildlife conservation, data analysis, or batch operations on images!
                            </Typography>
                        </Box>
                    ) : (
                        <>
                            {messages.map((message, index) => (
                                <ChatMessageRenderer
                                    key={message.id}
                                    message={message}
                                    onApplyConfirmation={handleApplyConfirmation}
                                    onRevertConfirmation={handleRevertConfirmation}
                                    isLoading={isLoading}
                                    isLastMessage={index === messages.length - 1}
                                />
                            ))}
                            {isLoading && messages[messages.length - 1]?.role !== 'assistant' && (
                                <MessageSkeleton />
                            )}
                        </>
                    )}
                </Container>
            </Box>

            {/* Input Bar */}
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
                    isLoading={isLoading}
                    onStopGeneration={handleStopGeneration}
                    onNewChat={handleNewChat}
                />
            </Box>


        </Box>
    );
};

export default AgentPage;
