import { Box, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Typography, useTheme } from '@mui/material'
import DashboardIcon from '@mui/icons-material/Dashboard'
import PhotoLibraryIcon from '@mui/icons-material/PhotoLibrary'
import AnalyticsIcon from '@mui/icons-material/Analytics'
import SettingsIcon from '@mui/icons-material/Settings'

const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon /> },
    { text: 'Library', icon: <PhotoLibraryIcon /> },
    { text: 'Analysis', icon: <AnalyticsIcon /> },
    { text: 'Settings', icon: <SettingsIcon /> },
]

export default function Sidebar() {
    const theme = useTheme()

    return (
        <Box sx={{
            width: 240,
            height: '100%',
            borderRight: `1px solid ${theme.palette.divider}`,
            bgcolor: 'background.paper',
            display: 'flex',
            flexDirection: 'column'
        }}>
            <Box sx={{ p: 3 }}>
                <Typography variant="h6" fontWeight="bold" color="primary">
                    RewildID Pro
                </Typography>
            </Box>

            <List sx={{ px: 2 }}>
                {menuItems.map((item) => (
                    <ListItem key={item.text} disablePadding sx={{ mb: 1 }}>
                        <ListItemButton sx={{ borderRadius: 2 }}>
                            <ListItemIcon sx={{ minWidth: 40, color: 'inherit' }}>
                                {item.icon}
                            </ListItemIcon>
                            <ListItemText primary={item.text} primaryTypographyProps={{ fontWeight: 500 }} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Box>
    )
}
