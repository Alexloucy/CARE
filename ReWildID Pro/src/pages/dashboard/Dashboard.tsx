import { Box, Typography } from '@mui/material';
import AiModeButton from '../../components/AiModeButton';

export default function Dashboard() {
    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
                Dashboard
            </Typography>
            <Typography variant="body1" sx={{ mb: 4 }}>
                Welcome to RewildID Pro.
            </Typography>

            <Box sx={{ display: 'flex', gap: 2 }}>
                <AiModeButton />
            </Box>
        </Box>
    );
}
