export interface FileDetails {
    name: string;
    isDirectory: boolean;
    path: string;
    parent?: string;
}

export interface BrowseImageResponse {
    ok: boolean;
    status?: number;
    files?: FileDetails[];
    error?: string;
}

export interface ViewImageResponse {
    ok: boolean;
    data?: Uint8Array;
    error?: string;
}

export interface DBImage {
    id: number;
    group_id: number;
    original_path: string;
    preview_path?: string;
    date_added: number;
    group_name: string;
    group_created_at: number;
}

export interface ElectronApi {
    browseImage: (date: string, folderPath: string) => Promise<BrowseImageResponse>;
    viewImage: (originalPath: string) => Promise<ViewImageResponse>;
    getImagePaths: (currentFolder: string) => Promise<{ ok: boolean; selectAllPaths?: string[]; error?: string }>;
    getImages: (filter?: { date?: string, groupIds?: number[], searchQuery?: string }) => Promise<{ ok: boolean; images?: DBImage[]; error?: string }>;
    downloadSelectedGalleryImages: (selectedPaths: string[]) => Promise<{ ok: boolean; error?: string }>;
    uploadImage: (relativePath: string, originalPath: string) => Promise<{ ok: boolean; error?: string }>;
    uploadPaths: (filePaths: string[], groupName?: string) => Promise<{ ok: boolean; count?: number; errors?: string[]; error?: string }>;
    deleteGroup: (id: number) => Promise<{ ok: boolean; error?: string }>;
    deleteImage: (id: number) => Promise<{ ok: boolean; error?: string }>;
    updateGroupName: (id: number, name: string) => Promise<{ ok: boolean; error?: string }>;
    checkIsDirectory: (filePath: string) => Promise<boolean>;
    openFileDialog: () => Promise<{ canceled: boolean; filePaths: string[] }>;
    saveImages: (sourcePaths: string[]) => Promise<{ ok: boolean; successCount?: number; failCount?: number; error?: string }>;
    detect: (selectedPaths: string[], onStream: (txt: string) => void) => Promise<{ ok: boolean; error?: string }>;
    browseDetectImage: (date: string, folderPath: string, filterLabel: string, confLow: number, confHigh: number) => Promise<BrowseImageResponse>;
    viewDetectImage: (date: string, imagePath: string) => Promise<ViewImageResponse>;
    getDetectImagePaths: (dirPath: string, filterLabel: string, confLow: number, confHigh: number) => Promise<{ ok: boolean; selectAllPaths?: string[]; error?: string }>;
    downloadDetectImages: (filterLabel: string) => Promise<{ ok: boolean; error?: string }>;
    downloadSelectedDetectImages: (selectPaths: string[]) => Promise<{ ok: boolean; error?: string }>;
    runReid: (selectedPaths: string[], onStream: (txt: string) => void) => Promise<{ ok: boolean; error?: string }>;
    browseReidImage: (date: string, time: string, group_id: string) => Promise<any>;
    downloadReidImages: (date: string, time: string) => Promise<{ ok: boolean; error?: string }>;
    deleteReidResult: (date: string, time: string) => Promise<{ ok: boolean; error?: string }>;
    renameReidGroup: (date: string, time: string, old_group_id: string, new_group_id: string) => Promise<{ ok: boolean; error?: string }>;
    terminateAI: () => Promise<void>;
    getPathForFile: (file: File) => string;
    getJobs: () => Promise<any[]>;
    cancelJob: (id: string) => Promise<void>;
    onJobUpdate: (callback: (jobs: any[]) => void) => () => void;
}

declare global {
    interface Window {
        api: ElectronApi;
    }
}
