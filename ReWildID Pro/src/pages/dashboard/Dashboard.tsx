import { Box, Typography, alpha, useTheme, Skeleton } from '@mui/material';
import { useState, useEffect } from 'react';
import { 
    Images, Sparkle, Fingerprint, FolderSimple, 
    ListBullets, Clock, FolderOpen, Tag
} from '@phosphor-icons/react';

interface DashboardStats {
    totalImages: number;
    totalGroups: number;
    totalDetections: number;
    totalSpecies: number;
    totalReidRuns: number;
    totalIndividuals: number;
    recentActivity: { type: string; name: string; count: number; date: number }[];
}

// Stat Card Component
const StatCard = ({ 
    title, 
    value, 
    icon: Icon, 
    color,
    loading 
}: { 
    title: string; 
    value: number | string; 
    icon: React.ElementType; 
    color: string;
    loading?: boolean;
}) => {
    return (
        <Box sx={{ 
            p: 2.5, 
            borderRadius: 3, 
            bgcolor: alpha(color, 0.08),
            border: `1px solid ${alpha(color, 0.2)}`,
            flex: 1,
            minWidth: 140,
            transition: 'all 0.2s',
            '&:hover': {
                bgcolor: alpha(color, 0.12),
                borderColor: alpha(color, 0.3)
            }
        }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1.5 }}>
                <Typography variant="body2" color="text.secondary" fontWeight={500}>
                    {title}
                </Typography>
                <Box sx={{ 
                    p: 0.75, 
                    borderRadius: 1.5, 
                    bgcolor: alpha(color, 0.15),
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}>
                    <Icon size={18} weight="duotone" color={color} />
                </Box>
            </Box>
            {loading ? (
                <Skeleton variant="text" width={60} height={40} />
            ) : (
                <Typography variant="h4" fontWeight={700} sx={{ color }}>
                    {typeof value === 'number' ? value.toLocaleString() : value}
                </Typography>
            )}
        </Box>
    );
};

// Activity Item Component
const ActivityItem = ({ 
    type, 
    name, 
    count, 
    date 
}: { 
    type: string; 
    name: string; 
    count: number; 
    date: number;
}) => {
    const theme = useTheme();
    const typeConfig: Record<string, { icon: React.ElementType; color: string; label: string }> = {
        group: { icon: FolderOpen, color: '#4285F4', label: 'Uploaded' },
        classification: { icon: Sparkle, color: '#9C27B0', label: 'Classified' },
        reid: { icon: Fingerprint, color: '#FF6B6B', label: 'Re-identified' }
    };
    const config = typeConfig[type] || typeConfig.group;
    const Icon = config.icon;
    
    const formatDate = (ts: number) => {
        const date = new Date(ts);
        const now = new Date();
        const diff = now.getTime() - date.getTime();
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        
        if (days === 0) return 'Today';
        if (days === 1) return 'Yesterday';
        if (days < 7) return `${days} days ago`;
        return date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
    };

    return (
        <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 2, 
            p: 1.5, 
            borderRadius: 2,
            '&:hover': { bgcolor: alpha(theme.palette.text.primary, 0.03) }
        }}>
            <Box sx={{ 
                p: 1, 
                borderRadius: 2, 
                bgcolor: alpha(config.color, 0.1),
                display: 'flex'
            }}>
                <Icon size={20} weight="duotone" color={config.color} />
            </Box>
            <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="body2" fontWeight={600} noWrap>{name}</Typography>
                <Typography variant="caption" color="text.secondary">
                    {config.label} • {count} item{count !== 1 ? 's' : ''}
                </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                {formatDate(date)}
            </Typography>
        </Box>
    );
};

export default function Dashboard() {
    const theme = useTheme();
    const [stats, setStats] = useState<DashboardStats | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadStats = async () => {
            try {
                const res = await window.api.getDashboardStats();
                if (res.ok && res.stats) {
                    setStats(res.stats);
                }
            } catch (e) {
                console.error('Failed to load dashboard stats:', e);
            }
            setLoading(false);
        };
        loadStats();
    }, []);

    return (
        <Box sx={{ pt: '64px', px: 3, pb: 3, minHeight: '100vh' }}>
            {/* Header */}
            <Box sx={{ py: 2, mb: 2 }}>
                <Typography variant="h5" fontWeight={600}>Dashboard</Typography>
                <Typography variant="body2" color="text.secondary">
                    Welcome to RewildID Pro
                </Typography>
            </Box>

            {/* Stats Grid */}
            <Box sx={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', 
                gap: 2, 
                mb: 4 
            }}>
                <StatCard 
                    title="Total Images" 
                    value={stats?.totalImages || 0} 
                    icon={Images} 
                    color="#4285F4"
                    loading={loading}
                />
                <StatCard 
                    title="Groups" 
                    value={stats?.totalGroups || 0} 
                    icon={FolderSimple} 
                    color="#34A853"
                    loading={loading}
                />
                <StatCard 
                    title="Detections" 
                    value={stats?.totalDetections || 0} 
                    icon={ListBullets} 
                    color="#9C27B0"
                    loading={loading}
                />
                <StatCard 
                    title="Species" 
                    value={stats?.totalSpecies || 0} 
                    icon={Tag} 
                    color="#FF9800"
                    loading={loading}
                />
                <StatCard 
                    title="ReID Runs" 
                    value={stats?.totalReidRuns || 0} 
                    icon={Sparkle} 
                    color="#E91E63"
                    loading={loading}
                />
                <StatCard 
                    title="Individuals" 
                    value={stats?.totalIndividuals || 0} 
                    icon={Fingerprint} 
                    color="#FF6B6B"
                    loading={loading}
                />
            </Box>

            {/* Recent Activity */}
            <Box sx={{ 
                p: 3, 
                borderRadius: 3, 
                border: `1px solid ${theme.palette.divider}`,
                bgcolor: theme.palette.mode === 'light' ? '#F9F9F9' : theme.palette.background.paper
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
                    <Clock size={22} weight="duotone" />
                    <Typography variant="h6" fontWeight={600}>Recent Activity</Typography>
                </Box>
                
                {loading ? (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        {[1, 2, 3].map(i => (
                            <Skeleton key={i} variant="rounded" height={56} />
                        ))}
                    </Box>
                ) : stats?.recentActivity && stats.recentActivity.length > 0 ? (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        {stats.recentActivity.map((activity, idx) => (
                            <ActivityItem 
                                key={idx} 
                                type={activity.type} 
                                name={activity.name} 
                                count={activity.count} 
                                date={activity.date} 
                            />
                        ))}
                    </Box>
                ) : (
                    <Box sx={{ py: 4, textAlign: 'center' }}>
                        <Clock size={48} weight="thin" color={theme.palette.text.disabled} />
                        <Typography color="text.secondary" sx={{ mt: 1 }}>
                            No recent activity yet
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            Start by uploading images in the Library
                        </Typography>
                    </Box>
                )}
            </Box>
        </Box>
    );
}
