import { useState } from 'react';

export function useLibraryUpload() {
    const [groupNameDialogOpen, setGroupNameDialogOpen] = useState(false);
    const [pendingUploadFiles, setPendingUploadFiles] = useState<string[]>([]);

    const processUploadPaths = async (paths: string[]) => {
        // Check if all paths are directories
        const areAllDirectories = await Promise.all(paths.map(path => window.api.checkIsDirectory(path)));
        const allDirs = areAllDirectories.every(isDir => isDir);

        if (allDirs) {
            // If all are directories, upload directly
            try {
                const result = await window.api.uploadPaths(paths);
                if (result.ok) {
                    // Import started
                } else {
                    console.error('Upload failed:', result.error);
                    alert('Upload failed: ' + result.error);
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert('Upload error occurred.');
            }
        } else {
            // If any are files, prompt for group name
            setPendingUploadFiles(paths);
            setGroupNameDialogOpen(true);
        }
    };

    const handleUploadClick = async () => {
        const result = await window.api.openFileDialog();
        if (!result.canceled && result.filePaths.length > 0) {
            processUploadPaths(result.filePaths);
        }
    };

    const handleConfirmUpload = async (name: string) => {
        setGroupNameDialogOpen(false);
        try {
            const response = await window.api.uploadPaths(pendingUploadFiles, name);
            if (response.ok) {
                // Import started
            } else {
                console.error('Upload failed:', response.error);
            }
        } catch (error) {
            console.error('Error uploading:', error);
        } finally {
            setPendingUploadFiles([]);
        }
    };

    return {
        groupNameDialogOpen,
        setGroupNameDialogOpen,
        pendingUploadFiles,
        setPendingUploadFiles,
        handleUploadClick,
        processUploadPaths,
        handleConfirmUpload
    };
}
