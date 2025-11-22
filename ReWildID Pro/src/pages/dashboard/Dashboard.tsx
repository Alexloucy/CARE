import { Box, Typography } from '@mui/material';

export default function Dashboard() {
    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
                Dashboard
            </Typography>
            <Typography variant="body1">
                Welcome to RewildID Pro.
            </Typography>
        </Box>
    );
}
