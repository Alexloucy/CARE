import React from 'react';
import { Box, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage } from '../../types/agent';
import CodeExecutionBlock from './CodeExecutionBlock';

interface ChatMessageRendererProps {
    message: ChatMessage;
}

const ChatMessageRenderer: React.FC<ChatMessageRendererProps> = ({ message }) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const isUser = message.role === 'user';
    const isTool = message.role === 'tool';
    const isAssistantWithToolCalls = message.role === 'assistant' && message.toolCalls && message.toolCalls.length > 0;

    // Get glass style based on message type
    const getGlassStyle = () => {
        if (isUser) {
            return {
                background: isDark
                    ? 'rgba(100, 149, 237, 0.2)'
                    : 'rgba(70, 130, 180, 0.15)',
                border: `1px solid ${isDark ? 'rgba(100, 149, 237, 0.3)' : 'rgba(70, 130, 180, 0.25)'}`,
            };
        }
        if (isTool) {
            // Tool result - show based on success/failure
            const isError = message.toolResult && !message.toolResult.success;
            if (isError) {
                return {
                    background: isDark
                        ? 'rgba(244, 67, 54, 0.15)'
                        : 'rgba(244, 67, 54, 0.1)',
                    border: `1px solid ${isDark ? 'rgba(244, 67, 54, 0.3)' : 'rgba(244, 67, 54, 0.25)'}`,
                };
            }
            return {
                background: isDark
                    ? 'rgba(76, 175, 80, 0.15)'
                    : 'rgba(76, 175, 80, 0.1)',
                border: `1px solid ${isDark ? 'rgba(76, 175, 80, 0.3)' : 'rgba(76, 175, 80, 0.25)'}`,
            };
        }
        if (isAssistantWithToolCalls) {
            // Tool call request - amber tint
            return {
                background: isDark
                    ? 'rgba(255, 193, 7, 0.15)'
                    : 'rgba(255, 193, 7, 0.1)',
                border: `1px solid ${isDark ? 'rgba(255, 193, 7, 0.3)' : 'rgba(255, 193, 7, 0.25)'}`,
            };
        }
        // Regular assistant message
        return {
            background: isDark
                ? 'rgba(255, 255, 255, 0.08)'
                : 'rgba(0, 0, 0, 0.08)',
            border: `1px solid ${isDark ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.12)'}`,
        };
    };

    const glassStyle = getGlassStyle();

    // Render tool call request (assistant asking to use a tool)
    if (isAssistantWithToolCalls) {
        return (
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: 'flex-start',
                    width: '100%',
                    mb: 2,
                }}
            >
                <Box
                    sx={{
                        maxWidth: '75%',
                        p: 1.5,
                        borderRadius: '16px 16px 16px 4px',
                        ...glassStyle,
                        backdropFilter: 'blur(76px)',
                        WebkitBackdropFilter: 'blur(76px)',
                    }}
                >
                    {message.toolCalls!.map((tc, idx) => (
                        <Typography
                            key={idx}
                            variant="body2"
                            sx={{
                                color: isDark ? 'rgba(255, 193, 7, 0.9)' : 'rgba(180, 140, 0, 1)',
                                fontStyle: 'italic',
                                fontSize: '0.85rem',
                            }}
                        >
                            🔧 Calling tool: {tc.name}
                        </Typography>
                    ))}
                </Box>
            </Box>
        );
    }

    // Render tool result
    if (isTool && message.toolResult) {
        return (
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: 'flex-start',
                    width: '100%',
                    mb: 2,
                }}
            >
                <Box
                    sx={{
                        maxWidth: '85%',
                        minWidth: '300px',
                    }}
                >
                    <CodeExecutionBlock
                        result={message.toolResult}
                        toolName={message.toolName || 'Tool'}
                    />
                </Box>
            </Box>
        );
    }

    // Render regular message (user or assistant text)
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
                    backdropFilter: 'blur(76px)',
                    WebkitBackdropFilter: 'blur(76px)',
                }}
            >
                {isUser ? (
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
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                        </ReactMarkdown>
                    </Box>
                )}
            </Box>
        </Box>
    );
};

export default ChatMessageRenderer;
