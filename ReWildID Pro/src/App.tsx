import { Routes, Route } from 'react-router-dom';
import Layout from './app/layout/Layout';
import Dashboard from './pages/dashboard/Dashboard';
import { Box, Typography } from '@mui/material';

const Placeholder = ({ title }: { title: string }) => (
    <Box sx={{ p: 3 }}>
        <Typography variant="h4">{title}</Typography>
    </Box>
);

function App() {
    return (
        <Routes>
            <Route path="/" element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route path="import" element={<Placeholder title="Import" />} />
                <Route path="library" element={<Placeholder title="Library" />} />
                <Route path="analysis" element={<Placeholder title="Analysis" />} />
                <Route path="settings" element={<Placeholder title="Settings" />} />
                <Route path="auth" element={<Placeholder title="Auth" />} />
                <Route path="*" element={<Placeholder title="404 Not Found" />} />
            </Route>
        </Routes>
    );
}

export default App;
