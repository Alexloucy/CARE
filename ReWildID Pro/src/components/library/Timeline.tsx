import React, { useEffect, useRef, useState } from 'react';
import { Box, Tooltip, useTheme, Typography } from '@mui/material';
import { DateSection } from '../../types/library';

interface TimelineProps {
    dateSections: DateSection[];
    onDateClick: (date: string) => void;
    onGroupClick: (groupId: number) => void;
    activeId?: string;
}

export const Timeline: React.FC<TimelineProps> = ({ dateSections, onDateClick, onGroupClick, activeId }) => {
    const theme = useTheme();
    const scrollRef = useRef<HTMLDivElement>(null);
    const activeRef = useRef<HTMLDivElement>(null);
    const [showLabels, setShowLabels] = useState(false);

    const formatDate = (dateStr: string) => {
        if (dateStr.length !== 8) return dateStr;
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    };

    // Flatten items but keep structure to calculate positions linearly
    const timelineItems = React.useMemo(() => {
        const items: Array<{ type: 'date' | 'group', id: string, label: string, onClick: () => void }> = [];
        dateSections.forEach(section => {
            items.push({
                type: 'date',
                id: `date-${section.date}`,
                label: formatDate(section.date),
                onClick: () => onDateClick(section.date)
            });
            section.groups.forEach(group => {
                items.push({
                    type: 'group',
                    id: `group-${group.id}`,
                    label: `${group.name} (${group.images.length})`,
                    onClick: () => onGroupClick(group.id)
                });
            });
        });
        return items;
    }, [dateSections, onDateClick, onGroupClick]);

    // Auto-scroll timeline to keep active item in view
    useEffect(() => {
        if (activeId && activeRef.current && scrollRef.current) {
            const activeEl = activeRef.current;
            const container = scrollRef.current;
            const activeRect = activeEl.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();

            // Calculate relative position
            const relativeTop = activeRect.top - containerRect.top;
            const relativeBottom = activeRect.bottom - containerRect.top;

            // Scroll if out of bounds (with some padding)
            if (relativeTop < 50) {
                container.scrollBy({ top: relativeTop - 100, behavior: 'smooth' });
            } else if (relativeBottom > containerRect.height - 50) {
                container.scrollBy({ top: relativeBottom - containerRect.height + 100, behavior: 'smooth' });
            }
        }
    }, [activeId]);

    if (timelineItems.length === 0) return null;

    const ITEM_SPACING = 35; // Minimum spacing in px
    const TOTAL_HEIGHT = Math.max(timelineItems.length * ITEM_SPACING, 100); // Ensure min height

    return (
        <Box sx={{ 
            position: 'fixed', 
            right: 0, 
            top: 160, 
            bottom: 40,
            width: 140, // Slightly narrower to move closer to right
            zIndex: 20,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-end',
            pointerEvents: 'none', 
            maskImage: 'linear-gradient(to bottom, transparent 0%, black 10%, black 90%, transparent 100%)',
            WebkitMaskImage: 'linear-gradient(to bottom, transparent 0%, black 10%, black 90%, transparent 100%)',
        }}>
            {/* Scrollable Container */}
            <Box 
                ref={scrollRef}
                sx={{
                    width: '100%',
                    height: '100%',
                    overflowY: 'auto',
                    overflowX: 'hidden',
                    position: 'relative',
                    '&::-webkit-scrollbar': { display: 'none' },
                    scrollbarWidth: 'none',
                    msOverflowStyle: 'none',
                    scrollBehavior: 'smooth',
                    pointerEvents: 'none', // Pass through clicks in scroll area
                    // Add padding to top/bottom so first/last items aren't masked
                    pt: 4,
                    pb: 4
                }}
            >
                <Box sx={{ position: 'relative', width: '100%', minHeight: '100%', height: `${TOTAL_HEIGHT}px` }}>
                    
                    {/* Interactive Area Wrapper */}
                    <Box 
                        sx={{ 
                            position: 'absolute', 
                            top: 0, 
                            right: -3, 
                            bottom: 0, 
                            width: 40,
                            pointerEvents: 'auto' // Enable interaction only on the strip
                        }}
                        onMouseEnter={() => setShowLabels(true)}
                        onMouseLeave={() => setShowLabels(false)}
                    >
                        {/* Vertical Line */}
                        <Box sx={{
                            position: 'absolute',
                            top: 0,
                            bottom: 0,
                            left: '50%',
                            width: '2px',
                            bgcolor: theme.palette.divider,
                            transform: 'translateX(-50%)',
                            borderRadius: 1,
                            zIndex: 0
                        }} />

                        {timelineItems.map((item, index) => {
                            const isActive = activeId === item.id;
                            const top = index * ITEM_SPACING + (ITEM_SPACING / 2); 
                            
                            return (
                                <React.Fragment key={item.id}>
                                    {/* Date Label */}
                                    {item.type === 'date' && (
                                        <Typography
                                            className="timeline-date-label"
                                            variant="caption"
                                            sx={{
                                                position: 'absolute',
                                                top: `${top}px`,
                                                right: 32, // Positioned left of dot area
                                                transform: 'translateY(50%) translateX(0px)',
                                                whiteSpace: 'nowrap',
                                                bgcolor: 'rgba(0,0,0,0.85)',
                                                color: 'white',
                                                px: 1.5,
                                                py: 0.5,
                                                borderRadius: 1,
                                                opacity: showLabels ? 1 : 0,
                                                transition: 'all 0.2s ease',
                                                pointerEvents: 'none',
                                                fontWeight: 700,
                                                fontSize: '0.75rem',
                                                lineHeight: 1, // Fix vertical alignment
                                                zIndex: 10,
                                                boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                                                display: 'flex',
                                                alignItems: 'center'
                                            }}
                                        >
                                            {item.label}
                                        </Typography>
                                    )}

                                    {/* Dot */}
                                    <Tooltip title={item.type === 'group' ? item.label : ''} placement="left">
                                        <Box
                                            ref={isActive ? activeRef : null}
                                            onClick={item.onClick}
                                            sx={{
                                                position: 'absolute',
                                                top: `${top}px`,
                                                right: 0,
                                                width: 40, // Match interactive strip width
                                                height: 40,
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                cursor: 'pointer',
                                                zIndex: 2,
                                                pointerEvents: 'auto', // Ensure dots are clickable
                                                '&:hover .timeline-dot': { 
                                                    transform: 'scale(1.5)',
                                                    bgcolor: theme.palette.primary.main
                                                }
                                            }}
                                            onMouseEnter={() => setShowLabels(true)}
                                            onMouseLeave={() => setShowLabels(false)}
                                        >
                                            <Box
                                                className="timeline-dot"
                                                sx={{
                                                    width: item.type === 'date' ? 12 : 8,
                                                    height: item.type === 'date' ? 12 : 8,
                                                    borderRadius: '50%',
                                                    bgcolor: isActive ? theme.palette.primary.main : (item.type === 'date' ? theme.palette.text.secondary : theme.palette.action.disabled),
                                                    transition: 'transform 0.2s ease, background-color 0.2s ease',
                                                    border: `2px solid ${theme.palette.background.default}`,
                                                    transform: isActive ? 'scale(1.5)' : 'scale(1)',
                                                }}
                                            />
                                        </Box>
                                    </Tooltip>
                                </React.Fragment>
                            );
                        })}
                    </Box>
                </Box>
            </Box>
        </Box>
    );
};
