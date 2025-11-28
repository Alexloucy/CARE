// Utility to trigger navigation and refresh from anywhere in the app

// Trigger global upload dialog
export const triggerUpload = () => {
    window.dispatchEvent(new CustomEvent('trigger-upload'));
};

// Trigger data refresh on specific pages
export const triggerRefresh = (page: 'library' | 'classification' | 'reid') => {
    window.dispatchEvent(new CustomEvent('trigger-refresh', { detail: { page } }));
};
