import { Box, Breadcrumbs, Link, Typography } from '@mui/material';
import { Link as RouterLink, useLocation } from 'react-router-dom';

interface BreadcrumbProps {
    customItems?: { label: string; path: string }[];
}

export default function Breadcrumb({ customItems }: BreadcrumbProps) {
    const location = useLocation();
    const pathnames = location.pathname.split('/').filter((x) => x);

    return (
        <Box role="presentation" sx={{ ml: 2 }}>
            <Breadcrumbs aria-label="breadcrumb">
                <Link component={RouterLink} underline="hover" color="inherit" to="/">
                    Home
                </Link>
                {customItems ? (
                    customItems.map((item, index) => {
                        const isLast = index === customItems.length - 1;
                        return isLast ? (
                            <Typography color="text.primary" key={item.path}>{item.label}</Typography>
                        ) : (
                            <Link component={RouterLink} underline="hover" color="inherit" to={item.path} key={item.path}>
                                {item.label}
                            </Link>
                        );
                    })
                ) : (
                    pathnames.map((value, index) => {
                        const last = index === pathnames.length - 1;
                        const to = `/${pathnames.slice(0, index + 1).join('/')}`;

                        return last ? (
                            <Typography color="text.primary" key={to}>
                                {value.charAt(0).toUpperCase() + value.slice(1)}
                            </Typography>
                        ) : (
                            <Link component={RouterLink} underline="hover" color="inherit" to={to} key={to}>
                                {value.charAt(0).toUpperCase() + value.slice(1)}
                            </Link>
                        );
                    })
                )}
            </Breadcrumbs>
        </Box>
    );
}
