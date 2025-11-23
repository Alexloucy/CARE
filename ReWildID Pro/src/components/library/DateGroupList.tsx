import React from 'react';
import { Box, Typography, IconButton, useTheme } from '@mui/material';
import { DotsThreeVertical, UploadSimple } from '@phosphor-icons/react';
import ImageCard from '../ImageCard';
import { DateSection } from '../../types/library';
import { DBImage } from '../../types/electron';

interface DateGroupListProps {
    dateSections: DateSection[];
    imageUrls: Record<number, string>;
    loadImage: (image: DBImage) => void;
    isSelectionMode: boolean;
    selectedImageIds: Set<number>;
    onToggleSelection: (id: number) => void;
    onImageClick: (image: DBImage) => void;
    onMenuOpen: (event: React.MouseEvent<HTMLElement>, groupId: number) => void;
}

export const DateGroupList: React.FC<DateGroupListProps> = ({
    dateSections,
    imageUrls,
    loadImage,
    isSelectionMode,
    selectedImageIds,
    onToggleSelection,
    onImageClick,
    onMenuOpen
}) => {
    const theme = useTheme();

    const formatDate = (dateStr: string) => {
        if (dateStr.length !== 8) return dateStr;
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
        return date.toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    };

    if (dateSections.length === 0) {
        return (
            <Box
                sx={{
                    height: '60vh',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    opacity: 0.6
                }}
            >
                <UploadSimple size={64} color={theme.palette.text.primary} weight="thin" />
                <Typography variant="h5" fontWeight="500" sx={{ mt: 3, color: 'text.primary' }}>
                    No images yet
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
                    Drag and drop or click Upload to start
                </Typography>
            </Box>
        );
    }

    return (
        <Box>
            {dateSections.map((section) => (
                <Box key={section.date} sx={{ mb: 5, mt: 2 }}>
                    <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, color: theme.palette.text.secondary, textTransform: 'uppercase', letterSpacing: '0.5px', fontSize: '0.875rem' }}>
                        {formatDate(section.date)}
                    </Typography>

                    {section.groups.map(group => (
                        <Box key={group.id} sx={{ mb: 4 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                        {group.name}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary" sx={{ bgcolor: theme.palette.action.selected, px: 1, py: 0.5, borderRadius: 1 }}>
                                        {group.images.length}
                                    </Typography>
                                </Box>
                                <IconButton
                                    size="small"
                                    onClick={(e) => onMenuOpen(e, group.id)}
                                    sx={{ opacity: 0.6, '&:hover': { opacity: 1 } }}
                                >
                                    <DotsThreeVertical size={20} />
                                </IconButton>
                            </Box>

                            <Box sx={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                                gap: 2
                            }}>
                                {group.images.map((img) => {
                                    const fileDetails = {
                                        name: img.original_path.split(/[\\/]/).pop() || 'image.jpg',
                                        path: img.original_path,
                                        isDirectory: false
                                    };

                                    return (
                                        <Box key={img.id}>
                                            <ImageCard
                                                file={fileDetails}
                                                date={section.date}
                                                // @ts-ignore - Wrapper to match prop type if necessary
                                                loadImage={() => loadImage(img)}
                                                imageUrl={imageUrls[img.id]}
                                                onClick={() => onImageClick(img)}
                                                selectable={isSelectionMode}
                                                selected={selectedImageIds.has(img.id)}
                                                onToggleSelection={() => onToggleSelection(img.id)}
                                            />
                                        </Box>
                                    );
                                })}
                            </Box>
                        </Box>
                    ))}
                </Box>
            ))}
        </Box>
    );
};
