import { useEffect, useRef } from 'react';
import { driver, DriveStep } from 'driver.js';
import { useTheme } from '@mui/material';
import 'driver.js/dist/driver.css';
import '../styles/driver-theme.css';

// Tour step definitions for each page
const DASHBOARD_STEPS: DriveStep[] = [
    {
        element: '[data-tour="new-job"]',
        popover: {
            title: 'Welcome to ReWildID! 🎉',
            description: 'Click here to start a new job. You can also drag & drop folders anywhere in the app!',
            side: 'bottom',
            align: 'end',
        },
    },
    {
        element: '[data-tour="nav-dashboard"]',
        popover: {
            title: 'Dashboard',
            description: 'Dashboard shows your project overview - statistics, recent activity, and insights.',
            side: 'right',
            align: 'start',
        },
    },
    {
        element: '[data-tour="nav-library"]',
        popover: {
            title: 'Library',
            description: 'Library is your unified workspace for browsing images, running AI classification, and re-identifying individuals.',
            side: 'right',
            align: 'start',
        },
    },
    {
        element: '[data-tour="nav-classification"]',
        popover: {
            title: 'Classification',
            description: 'Classification is where AI detects and identifies species in your images.',
            side: 'right',
            align: 'start',
        },
    },
    {
        element: '[data-tour="nav-reid"]',
        popover: {
            title: 'Re-identification',
            description: 'Re-ID tracks individual animals across images using AI-powered recognition.',
            side: 'right',
            align: 'start',
        },
    },
];

const LIBRARY_STEPS: DriveStep[] = [
    {
        element: '[data-tour="library-filter"]',
        popover: {
            title: 'Filter',
            description: 'Filter images by date, groups, or other criteria.',
            side: 'bottom',
            align: 'center',
        },
    },
    {
        element: '[data-tour="library-select"]',
        popover: {
            title: 'Select Mode',
            description: 'Enter selection mode to select multiple images for batch operations. You can also long-press any image to start selecting!',
            side: 'bottom',
            align: 'center',
        },
    },
    {
        element: '[data-tour="library-sort"]',
        popover: {
            title: 'Sort Images',
            description: 'Sort images by species, individual (from Re-ID runs), or filename.',
            side: 'bottom',
            align: 'center',
        },
    },
    {
        element: '[data-tour="library-analyse"]',
        popover: {
            title: 'Analyse with AI ✨',
            description: 'Click Analyse on any group to run AI classification or re-identification.',
            side: 'left',
            align: 'start',
        },
    },
    {
        element: '[data-tour="library-grid"]',
        popover: {
            title: 'Zoom Previews 🔍',
            description: 'Use Ctrl + Scroll (or pinch on trackpad) to zoom in/out the image grid!',
            side: 'top',
            align: 'center',
        },
    },
];

// LocalStorage keys for tracking tour completion
const TOUR_KEYS = {
    dashboard: 'tour_completed_dashboard',
    library: 'tour_completed_library',
    classification: 'tour_completed_classification',
};

interface OnboardingTourProps {
    page: 'dashboard' | 'library' | 'classification';
}

export const OnboardingTour: React.FC<OnboardingTourProps> = ({ page }) => {
    const theme = useTheme();
    const driverRef = useRef<ReturnType<typeof driver> | null>(null);

    // Set data-theme attribute on body for CSS styling
    useEffect(() => {
        document.body.setAttribute('data-theme', theme.palette.mode);
        return () => {
            document.body.removeAttribute('data-theme');
        };
    }, [theme.palette.mode]);

    useEffect(() => {
        const tourKey = TOUR_KEYS[page];
        const hasCompletedTour = localStorage.getItem(tourKey);

        if (hasCompletedTour) return;

        // Get steps based on page
        let steps: DriveStep[];
        switch (page) {
            case 'dashboard':
                steps = DASHBOARD_STEPS;
                break;
            case 'library':
                steps = LIBRARY_STEPS;
                break;
            case 'classification':
                steps = LIBRARY_STEPS.slice(0, 3); // Reuse first 3 library steps
                break;
            default:
                return;
        }

        // Small delay to ensure DOM is ready
        const timer = setTimeout(() => {
            // Check if first target exists
            const firstTarget = document.querySelector(steps[0].element as string);
            if (!firstTarget) {
                localStorage.setItem(tourKey, 'true');
                return;
            }

            // Create driver instance
            driverRef.current = driver({
                showProgress: true,
                animate: true,
                allowClose: true,
                stagePadding: 8,
                stageRadius: 8,
                popoverClass: 'driverjs-theme',
                steps: steps,
                onDestroyStarted: () => {
                    localStorage.setItem(tourKey, 'true');
                    driverRef.current?.destroy();
                },
                onDestroyed: () => {
                    localStorage.setItem(tourKey, 'true');
                },
            });

            driverRef.current.drive();
        }, 800);

        return () => {
            clearTimeout(timer);
            if (driverRef.current) {
                driverRef.current.destroy();
            }
        };
    }, [page]);

    return null; // This component doesn't render anything
};

// Helper function to reset all tours (for settings page)
export const resetAllTours = () => {
    Object.values(TOUR_KEYS).forEach(key => {
        localStorage.removeItem(key);
    });
};

export default OnboardingTour;
