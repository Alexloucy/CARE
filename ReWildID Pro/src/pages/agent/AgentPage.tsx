import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Box, Typography, Container } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import AgentInputBar from '../../components/agent/AgentInputBar';
import ChatMessageRenderer from '../../components/agent/ChatMessageRenderer';
import MessageSkeleton from '../../components/agent/MessageSkeleton';
import { ChatMessage } from '../../types/agent';
import {
    streamAgentResponse,
    generateMessageId,
    getAgentSettings,
} from '../../services/agentService';

const AgentPage: React.FC = () => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [streamingContent, setStreamingContent] = useState('');
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    // Auto-scroll to bottom when new messages arrive or streaming content updates
    useEffect(() => {
        if (scrollContainerRef.current) {
            scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
    }, [messages, streamingContent]);

    const handleNewChat = useCallback(() => {
        setMessages([]);
        setStreamingContent('');
        setIsLoading(false);
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
    }, []);

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

                if (chunk.type === 'text_delta') {
                    // Accumulate streaming content
                    accumulatedContent += chunk.content;
                    setStreamingContent(accumulatedContent);
                }
            }

            // After streaming completes, add the full message
            if (accumulatedContent && !abortControllerRef.current?.signal.aborted) {
                const assistantMessage: ChatMessage = {
                    id: generateMessageId(),
                    role: 'assistant',
                    content: accumulatedContent,
                    timestamp: new Date(),
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

    return (
        <Box
            sx={{
                position: 'relative',
                height: '100vh',
                overflow: 'hidden',
            }}
        >
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
                                minHeight: '50vh',
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
