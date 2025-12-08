import { Routes, Route } from 'react-router-dom';
import Layout from './app/layout/Layout';
import Dashboard from './pages/dashboard/Dashboard';
import LibraryPage from './pages/library/LibraryPage';
import ClassificationPage from './pages/classification/ClassificationPage';
import ReIDPage from './pages/reid/ReIDPage';
import IndividualDetailView from './pages/reid/IndividualDetailView';
import SettingsPage from './pages/settings/SettingsPage';
import AgentPage from './pages/agent/AgentPage';
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
                <Route path="library" element={<LibraryPage />} />
                <Route path="classification" element={<ClassificationPage />} />
                <Route path="reid" element={<ReIDPage />} />
                <Route path="reid/run/:runId/individual/:individualId" element={<IndividualDetailView />} />
                <Route path="agent" element={<AgentPage />} />
                <Route path="settings" element={<SettingsPage />} />
                <Route path="auth" element={<Placeholder title="Auth" />} />
                <Route path="*" element={<Placeholder title="404 Not Found" />} />
            </Route>
        </Routes>
    );
}

export default App;
