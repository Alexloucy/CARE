// Utility to trigger the global upload dialog from anywhere in the app
export const triggerUpload = () => {
    window.dispatchEvent(new CustomEvent('trigger-upload'));
};
