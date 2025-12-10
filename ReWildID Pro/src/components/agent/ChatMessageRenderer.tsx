import React, { useState } from 'react';
import { Box, Typography, Button, Collapse } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage, ConfirmationRequest } from '../../types/agent';
import CodeExecutionBlock from './CodeExecutionBlock';
import AgentImageModal from './AgentImageModal';
import { CheckCircle, ArrowCounterClockwise, Warning, CaretDown, CaretRight, CircleNotch } from '@phosphor-icons/react';

// Inject spin animation keyframes once
if (typeof document !== 'undefined' && !document.getElementById('spin-keyframes')) {
    const style = document.createElement('style');
    style.id = 'spin-keyframes';
    style.textContent = `
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
}

interface ChatMessageRendererProps {
    message: ChatMessage;
    onApplyConfirmation?: (messageId: string, confirmationRequest: ConfirmationRequest | undefined) => Promise<void>;
    onRevertConfirmation?: (messageId: string, confirmationRequest: ConfirmationRequest | undefined) => Promise<void>;
    isLoading?: boolean;
    isLastMessage?: boolean;
}

const ChatMessageRenderer: React.FC<ChatMessageRendererProps> = ({
    message,
    onApplyConfirmation,
    onRevertConfirmation,
    isLoading,
    isLastMessage,
}) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const isUser = message.role === 'user';
    const isTool = message.role === 'tool';
    const isConfirmation = message.role === 'confirmation';
    const isAssistantWithToolCalls = message.role === 'assistant' && message.toolCalls && message.toolCalls.length > 0;
    const [toolCallExpanded, setToolCallExpanded] = useState(false);

    // Image modal state for user-attached images
    const [imageModalOpen, setImageModalOpen] = useState(false);
    const [imageModalIndex, setImageModalIndex] = useState(0);

    const handleImageClick = (index: number) => {
        setImageModalIndex(index);
        setImageModalOpen(true);
    };

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
        if (isConfirmation) {
            // Confirmation request - orange tint
            return {
                background: isDark
                    ? 'rgba(255, 152, 0, 0.15)'
                    : 'rgba(255, 152, 0, 0.1)',
                border: `1px solid ${isDark ? 'rgba(255, 152, 0, 0.3)' : 'rgba(255, 152, 0, 0.25)'}`,
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
                        cursor: 'pointer',
                    }}
                    onClick={() => setToolCallExpanded(!toolCallExpanded)}
                >
                    {/* Collapsed header */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CircleNotch
                            size={16}
                            weight="bold"
                            color={isDark ? 'rgba(255, 193, 7, 0.9)' : 'rgba(180, 140, 0, 1)'}
                            style={{
                                animation: isLastMessage && isLoading ? 'spin 1s linear infinite' : 'none',
                            }}
                        />
                        <Typography
                            variant="body2"
                            sx={{
                                color: isDark ? 'rgba(255, 193, 7, 0.9)' : 'rgba(180, 140, 0, 1)',
                                fontWeight: 500,
                                fontSize: '0.85rem',
                            }}
                        >
                            Analyzing
                        </Typography>
                        <Box sx={{ ml: 'auto' }}>
                            {toolCallExpanded ? (
                                <CaretDown size={14} color={isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.4)'} />
                            ) : (
                                <CaretRight size={14} color={isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.4)'} />
                            )}
                        </Box>
                    </Box>

                    {/* Expanded details */}
                    <Collapse in={toolCallExpanded}>
                        <Box sx={{ mt: 1, pl: 3 }}>
                            {message.toolCalls!.map((tc, idx) => (
                                <Box key={idx} sx={{ mb: 1 }}>
                                    <Typography
                                        variant="caption"
                                        sx={{
                                            color: isDark ? 'rgba(255, 193, 7, 0.8)' : 'rgba(180, 140, 0, 0.9)',
                                            fontFamily: 'monospace',
                                            fontWeight: 600,
                                            fontSize: '0.75rem',
                                        }}
                                    >
                                        {tc.name}
                                    </Typography>
                                    {tc.args && Object.keys(tc.args).length > 0 && (
                                        <Box
                                            component="pre"
                                            sx={{
                                                mt: 0.5,
                                                p: 1,
                                                borderRadius: 1,
                                                background: isDark ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.05)',
                                                maxHeight: 150,
                                                overflowY: 'auto',
                                                overflowX: 'auto',
                                                fontSize: '0.7rem',
                                                fontFamily: 'monospace',
                                                color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
                                                whiteSpace: 'pre',
                                                m: 0,
                                            }}
                                        >
                                            <code>
                                                {JSON.stringify(tc.args, null, 2)}
                                            </code>
                                        </Box>
                                    )}
                                </Box>
                            ))}
                        </Box>
                    </Collapse>
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

    // Render confirmation request
    if (isConfirmation && message.confirmationRequest) {
        const req = message.confirmationRequest;
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
                        p: 2,
                        borderRadius: '16px 16px 16px 4px',
                        ...glassStyle,
                        backdropFilter: 'blur(16px)',
                        WebkitBackdropFilter: 'blur(16px)',
                    }}
                >
                    {/* Status indicator */}
                    <Box
                        sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 0.75,
                            mb: 1,
                            color: req.status === 'applied'
                                ? (isDark ? 'rgba(76, 175, 80, 0.9)' : 'rgba(56, 142, 60, 1)')
                                : req.status === 'reverted'
                                    ? (isDark ? 'rgba(158, 158, 158, 0.9)' : 'rgba(97, 97, 97, 1)')
                                    : (isDark ? 'rgba(255, 152, 0, 0.9)' : 'rgba(200, 120, 0, 1)'),
                        }}
                    >
                        {req.status === 'applied' ? (
                            <CheckCircle size={18} weight="fill" />
                        ) : req.status === 'reverted' ? (
                            <ArrowCounterClockwise size={18} weight="bold" />
                        ) : (
                            <Warning size={18} weight="fill" />
                        )}
                        <Typography
                            variant="body2"
                            sx={{
                                fontWeight: 600,
                                color: 'inherit',
                            }}
                        >
                            {req.status === 'applied' ? 'Applied' : req.status === 'reverted' ? 'Reverted' : 'Pending Confirmation'}
                        </Typography>
                    </Box>

                    <Typography variant="body2" sx={{ mb: 1 }}>
                        {req.description}
                    </Typography>

                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
                        {req.affectedCount} image{req.affectedCount !== 1 ? 's' : ''} {req.status === 'applied' ? 'updated' : req.status === 'reverted' ? 'restored' : 'will be updated'}
                    </Typography>

                    {/* Action buttons */}
                    <Box sx={{ display: 'flex', gap: 1 }}>
                        {req.status === 'pending' && onApplyConfirmation && (
                            <Button
                                variant="contained"
                                size="small"
                                color="primary"
                                disabled={isLoading}
                                onClick={() => onApplyConfirmation(message.id, req)}
                                sx={{ textTransform: 'none', fontWeight: 500 }}
                            >
                                Confirm
                            </Button>
                        )}
                        {req.status === 'applied' && req.backupPath && onRevertConfirmation && (
                            <Button
                                variant="outlined"
                                size="small"
                                color="warning"
                                disabled={isLoading}
                                onClick={() => onRevertConfirmation(message.id, req)}
                                sx={{ textTransform: 'none', fontWeight: 500 }}
                            >
                                Revert Changes
                            </Button>
                        )}
                    </Box>
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
                    <Box>
                        {/* Display attached images */}
                        {message.images && message.images.length > 0 && (
                            <Box sx={{
                                display: 'flex',
                                gap: 1,
                                flexWrap: 'wrap',
                                mb: message.content ? 1.5 : 0
                            }}>
                                {message.images.map((imageDataUrl, idx) => (
                                    <Box
                                        key={idx}
                                        onClick={() => handleImageClick(idx)}
                                        sx={{
                                            width: 120,
                                            height: 120,
                                            borderRadius: '8px',
                                            overflow: 'hidden',
                                            border: `1px solid ${isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)'}`,
                                            cursor: 'pointer',
                                            transition: 'transform 0.2s, box-shadow 0.2s',
                                            '&:hover': {
                                                transform: 'scale(1.02)',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                                            },
                                        }}
                                    >
                                        <img
                                            src={imageDataUrl}
                                            alt={`Attached ${idx + 1}`}
                                            style={{
                                                width: '100%',
                                                height: '100%',
                                                objectFit: 'cover',
                                            }}
                                        />
                                    </Box>
                                ))}
                            </Box>
                        )}
                        {message.content && (
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
                        )}

                        {/* Image Modal for user-attached images */}
                        {message.images && message.images.length > 0 && (
                            <AgentImageModal
                                open={imageModalOpen}
                                onClose={() => setImageModalOpen(false)}
                                imageUrl={null}
                                images={message.images}
                                initialIndex={imageModalIndex}
                            />
                        )}
                    </Box>
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
