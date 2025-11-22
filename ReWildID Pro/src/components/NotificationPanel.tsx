import { Box, Typography, List, ListItem, Paper, useTheme } from '@mui/material'
import NotificationsIcon from '@mui/icons-material/Notifications'

export default function NotificationPanel() {
    const theme = useTheme()

    return (
        <Box sx={{
            width: 300,
            height: '100%',
            borderLeft: `1px solid ${theme.palette.divider}`,
            bgcolor: 'background.paper',
            p: 2
        }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <NotificationsIcon sx={{ mr: 1, color: 'text.secondary' }} />
                <Typography variant="h6" fontWeight="600">
                    Activity
                </Typography>
            </Box>

            <List>
                <ListItem disablePadding sx={{ mb: 2 }}>
                    <Paper sx={{ p: 2, width: '100%', bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)' }}>
                        <Typography variant="subtitle2" fontWeight="bold">Detection Complete</Typography>
                        <Typography variant="body2" color="text.secondary">Processed 150 images in "Camera Trap A"</Typography>
                        <Typography variant="caption" color="text.disabled" sx={{ mt: 1, display: 'block' }}>2 mins ago</Typography>
                    </Paper>
                </ListItem>
                <ListItem disablePadding>
                    <Paper sx={{ p: 2, width: '100%', bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)' }}>
                        <Typography variant="subtitle2" fontWeight="bold">Upload Started</Typography>
                        <Typography variant="body2" color="text.secondary">Importing 500 files...</Typography>
                        <Typography variant="caption" color="text.disabled" sx={{ mt: 1, display: 'block' }}>10 mins ago</Typography>
                    </Paper>
                </ListItem>
            </List>
        </Box>
    )
}
