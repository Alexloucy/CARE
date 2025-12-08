import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, keyframes } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage } from '../../types/agent';

// Fade in animation for streaming text
const fadeIn = keyframes`
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
`;

interface ChatMessageRendererProps {
    message: ChatMessage;
}

const ChatMessageRenderer: React.FC<ChatMessageRendererProps> = ({ message }) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const isUser = message.role === 'user';
    const isTool = message.role === 'tool';
    const isStreaming = message.isStreaming;

    // Track previous content length for streaming fade-in effect
    const prevContentLengthRef = useRef(0);
    const [displayParts, setDisplayParts] = useState<{ text: string; isNew: boolean }[]>([]);

    // Update display parts when content changes during streaming
    useEffect(() => {
        if (isStreaming && message.content) {
            const prevLength = prevContentLengthRef.current;
            const currentContent = message.content;

            if (currentContent.length > prevLength) {
                const oldPart = currentContent.slice(0, prevLength);
                const newPart = currentContent.slice(prevLength);

                setDisplayParts([
                    { text: oldPart, isNew: false },
                    { text: newPart, isNew: true },
                ]);

                // After animation, merge all as old
                const timer = setTimeout(() => {
                    prevContentLengthRef.current = currentContent.length;
                    setDisplayParts([{ text: currentContent, isNew: false }]);
                }, 200);

                return () => clearTimeout(timer);
            }
        } else {
            // Reset for non-streaming messages
            prevContentLengthRef.current = 0;
            setDisplayParts([]);
        }
    }, [message.content, isStreaming]);

    // Glassmorphic styles
    const getGlassStyle = () => {
        if (isUser) {
            return {
                background: isDark
                    ? 'rgba(100, 149, 237, 0.2)'  // Cornflower blue tint
                    : 'rgba(70, 130, 180, 0.15)', // Steel blue tint
                border: `1px solid ${isDark ? 'rgba(100, 149, 237, 0.3)' : 'rgba(70, 130, 180, 0.25)'}`,
            };
        }
        if (isTool) {
            return {
                background: isDark
                    ? 'rgba(255, 193, 7, 0.15)'  // Amber tint
                    : 'rgba(255, 193, 7, 0.1)',
                border: `1px solid ${isDark ? 'rgba(255, 193, 7, 0.3)' : 'rgba(255, 193, 7, 0.25)'}`,
            };
        }
        // Assistant - darker for readability
        return {
            background: isDark
                ? 'rgba(255, 255, 255, 0.08)'
                : 'rgba(0, 0, 0, 0.08)',
            border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.12)'}`,
        };
    };

    const glassStyle = getGlassStyle();

    // Render streaming content with fade-in effect
    const renderStreamingContent = () => {
        if (displayParts.length === 0) {
            return (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                </ReactMarkdown>
            );
        }

        return (
            <>
                {displayParts.map((part, idx) => (
                    <Box
                        key={idx}
                        component="span"
                        sx={{
                            display: 'inline',
                            animation: part.isNew ? `${fadeIn} 0.2s ease-out` : 'none',
                        }}
                    >
                        {part.text}
                    </Box>
                ))}
                {/* Blinking cursor */}
                <Box
                    component="span"
                    sx={{
                        display: 'inline-block',
                        width: '2px',
                        height: '1em',
                        backgroundColor: theme.palette.text.primary,
                        marginLeft: '2px',
                        animation: 'blink 1s infinite',
                        '@keyframes blink': {
                            '0%, 50%': { opacity: 1 },
                            '51%, 100%': { opacity: 0 },
                        },
                    }}
                />
            </>
        );
    };

    return (
        <Box
            sx={{
                display: 'flex',
                justifyContent: isUser ? 'flex-end' : 'flex-start',
                width: '100%',
                mb: 2,
            }}
        >
            <Box
                sx={{
                    maxWidth: '75%',
                    p: 2,
                    borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                    ...glassStyle,
                    backdropFilter: 'blur(12px)',
                    WebkitBackdropFilter: 'blur(12px)',
                    boxShadow: isDark
                        ? '0 4px 20px rgba(0, 0, 0, 0.3)'
                        : '0 4px 20px rgba(0, 0, 0, 0.1)',
                }}
            >
                {isTool ? (
                    <Typography
                        variant="body2"
                        sx={{
                            color: isDark ? 'rgba(255, 193, 7, 0.9)' : 'rgba(200, 150, 0, 1)',
                            fontStyle: 'italic',
                            fontSize: '0.85rem',
                        }}
                    >
                        🔧 {message.content}
                    </Typography>
                ) : isUser ? (
                    <Typography
                        variant="body1"
                        sx={{
                            color: theme.palette.text.primary,
                            whiteSpace: 'pre-wrap',
                            lineHeight: 1.6,
                        }}
                    >
                        {message.content}
                    </Typography>
                ) : (
                    <Box
                        sx={{
                            color: theme.palette.text.primary,
                            '& p': { margin: 0, mb: 1, '&:last-child': { mb: 0 } },
                            '& ul, & ol': { mt: 1, mb: 1, pl: 2.5 },
                            '& li': { mb: 0.5 },
                            '& code': {
                                background: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)',
                                padding: '2px 6px',
                                borderRadius: '4px',
                                fontSize: '0.875em',
                                fontFamily: 'monospace',
                            },
                            '& pre': {
                                background: isDark ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.05)',
                                padding: '12px',
                                borderRadius: '8px',
                                overflow: 'auto',
                                '& code': {
                                    background: 'none',
                                    padding: 0,
                                },
                            },
                            '& a': {
                                color: isDark ? '#90caf9' : '#1976d2',
                                textDecoration: 'none',
                                '&:hover': { textDecoration: 'underline' },
                            },
                            '& blockquote': {
                                borderLeft: `3px solid ${isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.2)'}`,
                                margin: '8px 0',
                                paddingLeft: '12px',
                                color: theme.palette.text.secondary,
                            },
                            lineHeight: 1.6,
                        }}
                    >
                        {isStreaming ? renderStreamingContent() : (
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {message.content}
                            </ReactMarkdown>
                        )}
                    </Box>
                )}

                {/* Show tool call results */}
                {message.toolCalls && message.toolCalls.length > 0 && (
                    <Box sx={{ mt: 1.5, pt: 1.5, borderTop: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}` }}>
                        {message.toolCalls.map((tc, idx) => (
                            <Typography
                                key={idx}
                                variant="caption"
                                sx={{
                                    display: 'block',
                                    color: theme.palette.text.secondary,
                                    fontFamily: 'monospace',
                                    fontSize: '0.75rem',
                                }}
                            >
                                Used: {tc.toolName} → {tc.result}
                            </Typography>
                        ))}
                    </Box>
                )}
            </Box>
        </Box>
    );
};

export default ChatMessageRenderer;
