"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.uploadImage = uploadImage;
exports.uploadPaths = uploadPaths;
exports.browseImage = browseImage;
exports.getImages = getImages;
exports.getImagesByIds = getImagesByIds;
exports.deleteGroup = deleteGroup;
exports.deleteImage = deleteImage;
exports.updateGroupName = updateGroupName;
exports.getImagePaths = getImagePaths;
exports.viewImage = viewImage;
exports.viewDetectImage = viewDetectImage;
exports.downloadReidImages = downloadReidImages;
exports.downloadSelectedGalleryImages = downloadSelectedGalleryImages;
exports.detect = detect;
exports.browseDetectImage = browseDetectImage;
exports.getDetectImagePaths = getDetectImagePaths;
exports.downloadDetectImages = downloadDetectImages;
exports.downloadSelectedDetectImages = downloadSelectedDetectImages;
exports.runReid = runReid;
exports.getDetectionBatches = getDetectionBatches;
exports.updateDetectionBatchName = updateDetectionBatchName;
exports.deleteDetectionBatch = deleteDetectionBatch;
exports.getDetectionsForBatch = getDetectionsForBatch;
exports.getAvailableSpecies = getAvailableSpecies;
exports.updateDetectionLabel = updateDetectionLabel;
exports.deleteDetection = deleteDetection;
exports.updateImageMetadata = updateImageMetadata;
exports.getImageMetadata = getImageMetadata;
exports.browseReidImage = browseReidImage;
exports.deleteReidResult = deleteReidResult;
exports.renameReidGroup = renameReidGroup;
exports.terminateAI = terminateAI;
exports.checkIsDirectory = checkIsDirectory;
exports.openFileDialog = openFileDialog;
exports.saveImages = saveImages;
exports.smartReID = smartReID;
exports.getReidRuns = getReidRuns;
exports.getReidRun = getReidRun;
exports.deleteReidRunById = deleteReidRunById;
exports.updateReidRunName = updateReidRunName;
exports.getReidResults = getReidResults;
exports.getReidResultsForImage = getReidResultsForImage;
exports.getReidResultsForImages = getReidResultsForImages;
exports.getLatestDetectionsForImages = getLatestDetectionsForImages;
exports.updateReidIndividualName = updateReidIndividualName;
exports.updateReidIndividualColor = updateReidIndividualColor;
exports.mergeReidIndividuals = mergeReidIndividuals;
exports.getDashboardStats = getDashboardStats;
exports.clearEmbeddingsCache = clearEmbeddingsCache;
const fs_extra_1 = __importDefault(require("fs-extra"));
const path_1 = __importDefault(require("path"));
const archiver_1 = __importDefault(require("archiver"));
const os_1 = __importDefault(require("os"));
const electron_1 = require("electron");
const mime_types_1 = require("mime-types");
const database_1 = require("./database");
const jobs_1 = require("./jobs");
const python_1 = require("./python");
// Migration Routine
async function migrateLegacyData() {
    try {
        const userIdFolder = '1';
        const baseDir = path_1.default.join(process.cwd(), 'data/image_uploaded', userIdFolder);
        if (!await fs_extra_1.default.pathExists(baseDir))
            return;
        const processDir = async (dir) => {
            const list = await fs_extra_1.default.readdir(dir);
            for (const file of list) {
                const filePath = path_1.default.join(dir, file);
                const stat = await fs_extra_1.default.stat(filePath);
                if (stat.isDirectory()) {
                    await processDir(filePath);
                }
                else if (file.endsWith('.json')) {
                    try {
                        const { originalPath } = await fs_extra_1.default.readJson(filePath);
                        if (await fs_extra_1.default.pathExists(originalPath)) {
                            const groups = database_1.DatabaseService.getAllGroups();
                            let legacyGroup = groups.find(g => g.name === 'Legacy Import');
                            let groupId;
                            if (!legacyGroup) {
                                groupId = database_1.DatabaseService.createGroup('Legacy Import');
                            }
                            else {
                                groupId = legacyGroup.id;
                            }
                            database_1.DatabaseService.addImage(groupId, originalPath);
                        }
                    }
                    catch (e) {
                        console.error(`Error importing legacy JSON ${filePath}:`, e);
                    }
                }
            }
        };
        await processDir(baseDir);
        console.log('Legacy migration completed.');
    }
    catch (error) {
        console.error('Migration failed:', error);
    }
}
// Run migration on load (async)
migrateLegacyData();
function getAppDataDir() {
    if (process.platform === 'win32') {
        let appDataPath = process.env.APPDATA || process.env.LOCALAPPDATA;
        if (appDataPath) {
            return path_1.default.join(appDataPath, 'ml4sg-care');
        }
    }
    return path_1.default.join(os_1.default.homedir(), '.ml4sg-care');
}
const userProfileDir = getAppDataDir();
async function uploadImage(relativePath, originalPath) {
    return { ok: false, error: 'Use uploadPaths instead.' };
}
async function uploadPaths(filePaths, groupName, afterAction, species) {
    try {
        // Check for immediate errors (empty, etc)
        if (filePaths.length === 0) {
            return { ok: false, error: 'No files selected' };
        }
        // Add to Job Queue with optional chained action
        jobs_1.JobManager.getInstance().addJob('import', {
            filePaths,
            groupName,
            afterAction,
            species
        });
        return { ok: true, count: 0, errors: [] }; // Count 0 indicates async start
    }
    catch (error) {
        return { ok: false, error: 'uploadPaths failed: ' + error };
    }
}
async function browseImage(date, folderPath) {
    try {
        const userIdFolder = '1';
        let baseDir, targetDir;
        if (!date) {
            baseDir = path_1.default.join(userProfileDir, 'data/image_uploaded', userIdFolder);
            targetDir = path_1.default.resolve(baseDir); // Resolve the full path
            fs_extra_1.default.ensureDirSync(targetDir);
        }
        else {
            baseDir = path_1.default.join(userProfileDir, 'data/image_uploaded', userIdFolder, date);
            targetDir = path_1.default.resolve(baseDir, folderPath); // Resolve the full path
            fs_extra_1.default.ensureDirSync(targetDir);
        }
        // Ensure the resolved path is still within the baseDir
        if (!targetDir.startsWith(baseDir)) {
            return { ok: false, error: 'Invalid folder path.' };
        }
        // Check if the directory exists before reading it
        if (!(await fs_extra_1.default.pathExists(targetDir))) {
            return { ok: false, error: 'Directory not found.' };
        }
        const stat = fs_extra_1.default.statSync(targetDir);
        if (stat.isFile()) {
            return { ok: false, error: 'Path is a file, not a directory.' };
        }
        const files = await fs_extra_1.default.readdir(targetDir);
        const fileDetails = await Promise.all(files.map(async (file) => {
            const filePath = path_1.default.join(targetDir, file);
            const stat = await fs_extra_1.default.stat(filePath);
            return {
                name: file,
                isDirectory: stat.isDirectory(),
                path: path_1.default.join(folderPath, file)
            };
        }));
        return { ok: true, status: 200, files: fileDetails };
    }
    catch (error) {
        return { ok: false, error: 'browseImage failed: ' + error };
    }
}
async function getImages(filter) {
    try {
        // Cleanup missing files first
        // DatabaseService.cleanupMissingImages(); // Optional: Enable if performance allows
        const images = database_1.DatabaseService.getImages(filter);
        return { ok: true, images };
    }
    catch (error) {
        return { ok: false, error: 'getImages failed: ' + error };
    }
}
async function getImagesByIds(imageIds) {
    try {
        const images = database_1.DatabaseService.getImagesByIds(imageIds);
        return { ok: true, images };
    }
    catch (error) {
        return { ok: false, error: 'getImagesByIds failed: ' + error };
    }
}
async function deleteGroup(id) {
    try {
        database_1.DatabaseService.deleteGroup(id);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'deleteGroup failed: ' + error };
    }
}
async function deleteImage(id) {
    try {
        database_1.DatabaseService.deleteImage(id);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'deleteImage failed: ' + error };
    }
}
async function updateGroupName(id, name) {
    try {
        database_1.DatabaseService.updateGroupName(id, name);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'updateGroupName failed: ' + error };
    }
}
// Deprecated / Modified for compatibility
async function getImagePaths(currentFolder) {
    const result = await getImages();
    // Map to expected legacy format if strictly needed, but we will update frontend.
    return { ok: true, selectAllPaths: [] }; // Return empty to force frontend update or avoid errors
}
async function viewImage(originalPath) {
    try {
        if (!originalPath) {
            return { ok: false, error: 'No file path provided.' };
        }
        if (!(await fs_extra_1.default.pathExists(originalPath))) {
            return { ok: false, error: 'File not found.' };
        }
        const data = await fs_extra_1.default.readFile(originalPath);
        return { ok: true, data };
    }
    catch (error) {
        return { ok: false, error: 'viewImage failed: ' + error };
    }
}
async function viewImageInPath(dir, date, imagePath) {
    let baseDir, targetDir;
    if (!date || !imagePath) {
        return { ok: false, error: 'Missing date or imagePath parameters.' };
    }
    else {
        const userIdFolder = '1';
        baseDir = path_1.default.join(process.cwd(), dir, userIdFolder, date);
        targetDir = path_1.default.resolve(baseDir, imagePath); // Resolve the full path
    }
    // Ensure the resolved path is still within the baseDir
    if (!targetDir.startsWith(baseDir)) {
        return { ok: false, error: 'Invalid folder path.' };
    }
    // Check for JSON reference first
    const jsonPath = targetDir + '.json';
    // Also check if the targetDir itself ends in .json (if the frontend requested the json file directly)
    const isJsonRequest = targetDir.endsWith('.json');
    const pathToCheck = isJsonRequest ? targetDir : jsonPath;
    if (await fs_extra_1.default.pathExists(pathToCheck)) {
        try {
            const { originalPath } = await fs_extra_1.default.readJson(pathToCheck);
            if (await fs_extra_1.default.pathExists(originalPath)) {
                const data = await fs_extra_1.default.readFile(originalPath);
                return { ok: true, data: data };
            }
            else {
                // Broken link
                await fs_extra_1.default.unlink(pathToCheck);
                return { ok: false, error: 'Original file not found.' };
            }
        }
        catch (e) {
            console.error(`Error reading JSON reference ${pathToCheck}:`, e);
        }
    }
    // Fallback: Check if the directory/file exists directly (Legacy support)
    if (!(await fs_extra_1.default.pathExists(targetDir))) {
        return { ok: false, error: 'File not found.' };
    }
    const stat = fs_extra_1.default.statSync(targetDir);
    if (stat.isDirectory()) {
        return { ok: false, error: 'Path is a directory, not an image file.' };
    }
    const mimeType = (0, mime_types_1.lookup)(targetDir);
    if (!mimeType || !mimeType.startsWith('image/')) {
        return { ok: false, error: 'File is not an image.' };
    }
    const data = await fs_extra_1.default.readFile(targetDir);
    return { ok: true, data: data };
}
async function viewDetectImage(date, imagePath) {
    try {
        return viewImageInPath('data/image_marked', date, imagePath);
    }
    catch (error) {
        return { ok: false, error: 'viewDetectImage failed: ' + error };
    }
}
async function saveZip(baseDir, selectedPaths, filename) {
    // Check if selectedPaths is an array and contains at least one file path
    if (!Array.isArray(selectedPaths) || selectedPaths.length === 0) {
        return { ok: false, error: 'No image paths provided.' };
    }
    // Ensure the files exist before archiving.
    if (!fs_extra_1.default.existsSync(baseDir)) {
        return { ok: false, error: 'Base source dir not found.' };
    }
    const result = await electron_1.dialog.showSaveDialog({
        title: 'Save archive as',
        filters: [{ name: 'Zip', extensions: ['zip'] }],
        defaultPath: filename
    });
    if (result.canceled) {
        return { ok: true };
    }
    const output = fs_extra_1.default.createWriteStream(result.filePath, { flush: true });
    const archive = (0, archiver_1.default)('zip', {
        zlib: { level: 0 } // Sets the compression level
    });
    // Pipe archive data to the response
    archive.pipe(output);
    // Append files to the archive while maintaining the folder structure
    for (const filePath of selectedPaths) {
        const fullPath = path_1.default.resolve(baseDir, filePath); // Resolve the full path
        try {
            // Check if file exists using fs-extra
            await fs_extra_1.default.access(fullPath);
            archive.file(fullPath, { name: filePath }); // Maintain folder structure in the archive
        }
        catch (err) {
            console.warn(`File not found: ${fullPath}`); // Log missing files
        }
    }
    // Finalize the archive (i.e., finish the zipping process)
    const endPromise = new Promise((resolve, _) => {
        output.on('finish', resolve);
    });
    const closePromise = new Promise((resolve, _) => {
        output.on('close', resolve);
    });
    await Promise.all([archive.finalize(), endPromise, closePromise]);
    return { ok: true };
}
async function downloadReidImages(date, time) {
    try {
        // Note: This sorts the images into folder by group ID. So we can't just
        // use saveZip here.
        const userIdFolder = '1';
        const baseDir = path_1.default.join(userProfileDir, 'data/image_reid_output', userIdFolder);
        const baseImgDir = path_1.default.join(userProfileDir, 'data/image_marked', userIdFolder);
        let targetDir;
        const timeJson = time + '.json';
        let relDir = path_1.default.join(date, timeJson);
        targetDir = path_1.default.resolve(baseDir, relDir);
        // Ensure the resolved path is still within the baseDir
        if (!targetDir.startsWith(baseDir)) {
            return { ok: false, error: 'Invalid folder path.' };
        }
        // Check if the directory exists before reading it
        if (!(await fs_extra_1.default.pathExists(targetDir))) {
            return { ok: false, error: 'Directory not found.' };
        }
        // Ensure the base directory exists
        fs_extra_1.default.ensureDirSync(baseDir);
        const result = await electron_1.dialog.showSaveDialog({
            title: 'Save archive as',
            filters: [{ name: 'Zip', extensions: ['zip'] }],
            defaultPath: `reid_images_${timestamp()}.zip`
        });
        if (result.canceled) {
            return { ok: true };
        }
        const output = fs_extra_1.default.createWriteStream(result.filePath);
        const archive = (0, archiver_1.default)('zip', {
            zlib: { level: 0 } // Sets the compression level
        });
        // Pipe archive data to the response
        archive.pipe(output);
        // Read and parse the JSON file
        const fileStructure = JSON.parse(fs_extra_1.default.readFileSync(targetDir, 'utf-8'));
        // Iterate through the folder (key) and files (value) in the JSON structure
        for (const [folder, files] of Object.entries(fileStructure)) {
            for (const filePath of files) {
                const fullPath = path_1.default.resolve(baseImgDir, filePath); // Resolve the full path
                try {
                    // Check if file exists using fs-extra
                    await fs_extra_1.default.access(fullPath);
                    const fileName = path_1.default.basename(filePath); // Extract file name
                    archive.file(fullPath, { name: path_1.default.join(folder, fileName) }); // Add file under the respective folder in the archive
                }
                catch (err) {
                    console.warn(`File not found: ${fullPath}`); // Log missing files
                }
            }
        }
        // Finalize the archive (i.e., finish the zipping process)
        await archive.finalize();
        await new Promise((resolve, _) => {
            output.close(resolve);
        });
        return { ok: true };
    }
    catch (error) {
        console.log(error);
        return { ok: false, error: 'downloadReidImages: ' + error };
    }
}
async function downloadSelectedGalleryImages(selectedPaths) {
    try {
        const userIdFolder = '1';
        const baseDir = path_1.default.join(userProfileDir, 'data/image_uploaded', userIdFolder);
        const filename = `gallery_images_${timestamp()}.zip`;
        return saveZip(baseDir, selectedPaths, filename);
    }
    catch (error) {
        return { ok: false, error: 'downloadSelectedGalleryImages failed: ' + error };
    }
}
async function detect(selectedPaths, stream, imageIds) {
    try {
        if (!selectedPaths || !Array.isArray(selectedPaths) || selectedPaths.length === 0) {
            return { ok: false, error: 'No images selected.' };
        }
        // Add to Job Queue with optional imageIds for caching
        jobs_1.JobManager.getInstance().addJob('detect', { selectedPaths, imageIds });
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'detect failed: ' + error };
    }
}
async function browseDetectImage(date, folderPath, filterLabel, confLow, confHigh) {
    try {
        const userIdFolder = '1';
        let baseDir, targetDir;
        if (!date) {
            baseDir = path_1.default.join(userProfileDir, 'data/image_marked', userIdFolder);
            targetDir = path_1.default.resolve(baseDir); // Resolve the full path
            fs_extra_1.default.ensureDirSync(targetDir);
        }
        else {
            baseDir = path_1.default.join(userProfileDir, 'data/image_marked', userIdFolder, date);
            targetDir = path_1.default.resolve(baseDir, folderPath); // Resolve the full path
            fs_extra_1.default.ensureDirSync(targetDir);
        }
        // Ensure the resolved path is still within the baseDir
        if (!targetDir.startsWith(baseDir)) {
            return { ok: false, error: 'Invalid folder path.' };
        }
        // Check if the directory exists before reading it
        if (!(await fs_extra_1.default.pathExists(targetDir))) {
            return { ok: false, error: 'Directory not found.' };
        }
        const stat = fs_extra_1.default.statSync(targetDir);
        if (stat.isFile()) {
            return { ok: false, error: 'Path is a file, not a directory.' };
        }
        const files = await fs_extra_1.default.readdir(targetDir);
        const fileDetails = await Promise.all(files.map(async (file) => {
            const filePath = path_1.default.join(targetDir, file);
            const stat = await fs_extra_1.default.stat(filePath);
            // Step 1: Check if it's a directory
            if (stat.isDirectory()) {
                return { name: file, isDirectory: true, path: path_1.default.join(folderPath, file) };
            }
            // Step 2: If it's not a directory, construct the corresponding JSON path
            // Assuming filePath points to the image file
            const relativeFilePath = path_1.default.relative(path_1.default.join(userProfileDir, 'data/image_marked'), filePath);
            // Extract the file name without the extension
            const fileNameWithoutExt = path_1.default.basename(relativeFilePath, path_1.default.extname(relativeFilePath));
            // Construct the corresponding JSON file path by replacing the base folder and appending `.json`
            const jsonFilePath = path_1.default.join(userProfileDir, 'data/image_cropped_json', path_1.default.dirname(relativeFilePath), // Keeps the directory structure intact
            `${fileNameWithoutExt}.json`);
            // Step 3: Extract label and confidence from the corresponding JSON file
            const jsonData = await extractLabelAndConfidence(jsonFilePath);
            if (!jsonData)
                return null; // Skip if the JSON cannot be read
            let predictions = jsonData;
            predictions = predictions.filter((p) => p.label !== 'blank'); // Filter out 'blank' labels
            console.log('predictions: ', predictions);
            // Step 4: Filtering logic based on the query parameters
            const isLabelNoDetection = filterLabel === 'No Detection'; // Check if filterLabel is the string "NoDetection"
            // Check if label represents "no detection" - can be null, empty array, or array containing 'blank'
            const labelIsNoDetection = !predictions ||
                predictions.length === 0 ||
                (predictions[0] && predictions[0].label === null) ||
                predictions.every((l) => l && l.label === 'blank');
            // Keep the prediction array when one of the predictions has the filterLabel
            const isLabelMatch = isLabelNoDetection
                ? labelIsNoDetection
                : !filterLabel ||
                    (Array.isArray(predictions)
                        ? predictions.some((l) => l && l.label === filterLabel)
                        : predictions && predictions.label === filterLabel);
            // Apply confidence filtering only if filterLabel is not "null"
            // keep the prediction array when one of the predictions has a confidence within the range
            const isConfidenceMatch = !isLabelNoDetection &&
                Array.isArray(predictions) &&
                predictions.some((l) => l &&
                    typeof l.pred_conf === 'number' &&
                    l.pred_conf >= confLow &&
                    l.pred_conf <= confHigh);
            if (isLabelMatch && (isLabelNoDetection || isConfidenceMatch)) {
                return { name: file, isDirectory: false, path: path_1.default.join(folderPath, file) };
            }
            return null; // Skip if the file doesn't match the filter
        }));
        // Filter out null values (files that didn't pass the filter)
        const filteredFiles = fileDetails.filter((file) => file !== null);
        return { ok: true, files: filteredFiles };
    }
    catch (error) {
        return { ok: false, error: 'browseDetectImages failed: ' + error };
    }
}
async function extractLabelAndConfidence(filePath) {
    try {
        // Use fs-extra to read and parse JSON directly
        const jsonData = await fs_extra_1.default.readJson(filePath);
        // Extract the predictions array from boxes
        const predictions = jsonData.boxes || [];
        return predictions; // Return array directly
    }
    catch (error) {
        console.error('Error reading or parsing the file:', error);
        return null; // If JSON cannot be read or parsed, return null to skip this file
    }
}
// Function to get all file paths
async function getDetectFilePaths(dir, baseDir, filterLabel, confLow, confHigh) {
    let results = [];
    const list = await fs_extra_1.default.readdir(dir);
    for (const file of list) {
        const filePath = path_1.default.join(dir, file);
        const stat = await fs_extra_1.default.stat(filePath);
        if (stat && stat.isDirectory()) {
            const subResults = await getDetectFilePaths(filePath, baseDir, filterLabel, confLow, confHigh);
            results = results.concat(subResults);
        }
        else {
            const relativeFilePath = path_1.default.relative(path_1.default.join(userProfileDir, 'data/image_marked'), filePath);
            // Extract the file name without the extension
            const fileNameWithoutExt = path_1.default.basename(relativeFilePath, path_1.default.extname(relativeFilePath));
            // Construct the corresponding JSON file path by replacing the base folder and appending `.json`
            const jsonFilePath = path_1.default.join(userProfileDir, 'data/image_cropped_json', path_1.default.dirname(relativeFilePath), // Keeps the directory structure intact
            `${fileNameWithoutExt}.json`);
            // Step 3: Extract label and confidence from the corresponding JSON file
            const predictions = await extractLabelAndConfidence(jsonFilePath);
            if (predictions) {
                // Step 4: Filtering logic based on the query parameters
                const isLabelNoDetection = filterLabel === 'No Detection'; // Check if filterLabel is the string "NoDetection"
                // Check if label represents "no detection" - can be null, empty array, or array containing only 'blank'
                const labelIsNoDetection = !predictions ||
                    predictions.length === 0 ||
                    (predictions[0] && predictions[0].label === null) ||
                    predictions.every((l) => l && l.label === 'blank');
                // keep the prediction array when one of the predictions has the filterLabel
                const isLabelMatch = isLabelNoDetection
                    ? labelIsNoDetection
                    : !filterLabel ||
                        (Array.isArray(predictions)
                            ? predictions.some((l) => l && l.label === filterLabel)
                            : predictions && predictions.label === filterLabel);
                // console.log('isLabelMatch: ', isLabelMatch)
                // console.log('filepath: ', filePath)
                // Apply confidence filtering only if filterLabel is not "null"
                // keep the prediction array when one of the predictions has a confidence within the range
                const isConfidenceMatch = !isLabelNoDetection &&
                    Array.isArray(predictions) &&
                    predictions.some((l) => l && l.confidence >= confLow && l.confidence <= confHigh);
                // console.log('isConfidenceMatch: ', isConfidenceMatch)
                if (isLabelMatch && (isLabelNoDetection || isConfidenceMatch)) {
                    const relativePath = path_1.default.relative(baseDir, filePath);
                    results.push(relativePath);
                }
            }
        }
    }
    return results;
}
async function getDetectImagePaths(dirPath, filterLabel, confLow, confHigh) {
    try {
        const userIdFolder = '1';
        const baseDir = path_1.default.join(userProfileDir, 'data/image_marked', userIdFolder);
        const targetDir = path_1.default.resolve(baseDir, dirPath); // Resolve the full path
        fs_extra_1.default.ensureDirSync(targetDir);
        // Ensure the resolved path is still within the baseDir
        if (!targetDir.startsWith(baseDir)) {
            return { ok: false, error: 'Invalid folder path.' };
        }
        // Check if the directory exists before reading it
        if (!(await fs_extra_1.default.pathExists(targetDir))) {
            return { ok: false, error: 'Directory not found.' };
        }
        const stat = fs_extra_1.default.statSync(targetDir);
        if (stat.isFile()) {
            return { ok: false, error: 'Path is a file, not a directory.' };
        }
        const filePaths = await getDetectFilePaths(targetDir, baseDir, filterLabel, confLow, confHigh);
        return { ok: true, selectAllPaths: filePaths };
    }
    catch (error) {
        console.log(error);
        return { ok: false, error: 'getDetectImagePaths failed: ' + error };
    }
}
async function downloadDetectImages(filterLabel) {
    try {
        const userIdFolder = '1';
        const baseDir = path_1.default.join(userProfileDir, 'data/image_marked', userIdFolder);
        fs_extra_1.default.ensureDirSync(baseDir);
        const filePaths = await getDetectFilePaths(baseDir, baseDir, filterLabel, 0, 1);
        const filename = `detection_${filterLabel}_images_${timestamp()}.zip`;
        return saveZip(baseDir, filePaths, filename);
    }
    catch (error) {
        console.log(error);
        return { ok: false, error: 'downloadDetectImages failed: ' + error };
    }
}
function timestamp() {
    // Generate timestamp-based zip filename using current timezone in YYYYMMDD_HHMMSS format
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0'); // Months are 0-based
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    return `${year}${month}${day}_${hours}${minutes}${seconds}`;
}
async function downloadSelectedDetectImages(selectPaths) {
    try {
        const userIdFolder = '1';
        const baseDir = path_1.default.join(userProfileDir, 'data/image_marked', userIdFolder);
        const filename = `detection_images_${timestamp()}.zip`;
        return saveZip(baseDir, selectPaths, filename);
    }
    catch (error) {
        console.log(error);
        return { ok: false, error: 'downloadSelectedDetectImages failed: ' + error };
    }
}
async function runReid(selectedPaths, stream) {
    const userIdFolder = '1';
    const tempImagePath = path_1.default.join(userProfileDir, 'temp/image_reid_pending', userIdFolder);
    const tempJsonPath = path_1.default.join(userProfileDir, 'temp/image_cropped_reid_pending', userIdFolder);
    try {
        (0, python_1.terminateSubprocess)();
        await fs_extra_1.default.remove(tempImagePath);
        await fs_extra_1.default.remove(tempJsonPath);
        if (!selectedPaths || !Array.isArray(selectedPaths) || selectedPaths.length === 0) {
            return { ok: false, error: 'No images selected or invalid format.' };
        }
        // Copy selected image to a temp folder for ReID
        for (const imagePath of selectedPaths) {
            const baseDir = path_1.default.join(userProfileDir, 'data/image_uploaded', userIdFolder);
            const srcPath = path_1.default.resolve(baseDir, imagePath); // Resolve the full path
            // Ensure the resolved path is still within the baseDir
            if (!srcPath.startsWith(baseDir)) {
                await fs_extra_1.default.remove(tempImagePath);
                await fs_extra_1.default.remove(tempJsonPath);
                return { ok: false, error: 'Invalid folder path.' };
            }
            const destPath = path_1.default.join(userProfileDir, 'temp/image_reid_pending', userIdFolder, imagePath);
            // Check if the source image exists
            if (await fs_extra_1.default.pathExists(srcPath)) {
                // Ensure the destination directory exists
                await fs_extra_1.default.ensureDir(path_1.default.dirname(destPath));
                // Copy the image
                await fs_extra_1.default.copy(srcPath, destPath);
            }
            else {
                console.warn(`runReid: File not found: ${imagePath}`);
            }
        }
        let args = [
            'reid',
            path_1.default.join(userProfileDir, 'temp/image_reid_pending', userIdFolder),
            path_1.default.join(userProfileDir, 'data/image_cropped_json', userIdFolder),
            path_1.default.join(userProfileDir, 'temp/image_cropped_reid_pending', userIdFolder),
            path_1.default.join(userProfileDir, 'data/image_reid_output', userIdFolder),
            path_1.default.join(userProfileDir, 'logs')
        ];
        let ps = (0, python_1.spawnPythonSubprocess)(args);
        if (!ps) {
            return { ok: false, error: 'Failed to start process' };
        }
        // Note: We track the process on a global, but only reference it in a local var, as another
        // event handler could clear the global var.
        (0, python_1.setSubProcess)(ps);
        if (ps.stdout) {
            ps.stdout.on('data', (data) => {
                console.log(`stdout: ${data}`);
                stream(data);
            });
        }
        if (ps.stderr) {
            ps.stderr.on('data', (data) => {
                console.log(`stderr: ${data}`);
                stream(data);
            });
        }
        return await new Promise((resolve, reject) => {
            ps.on('close', (code) => {
                console.log(`child process exited with code ${code}`);
                fs_extra_1.default.remove(tempImagePath);
                if (code != 0) {
                    reject({ ok: false, error: 'ERROR: Detection AI model error, please contact support.' });
                }
                (0, python_1.setSubProcess)(null);
                resolve({ ok: true });
            });
        });
    }
    catch (error) {
        return { ok: false, error: 'runReid failed: ' + error };
    }
    finally {
        await fs_extra_1.default.remove(tempImagePath);
    }
}
// Function to read the JSON file and extract keys
const extractKeysFromJson = async (filePath) => {
    try {
        // Read the JSON file
        const data = fs_extra_1.default.readFileSync(filePath, 'utf8');
        const jsonObject = JSON.parse(data);
        // Extract keys into a list
        return Object.keys(jsonObject);
    }
    catch (error) {
        console.error('Error reading or parsing JSON file:', error);
        return [];
    }
};
// ... existing code ...
// --- Detection Batches ---
async function getDetectionBatches() {
    try {
        const batches = database_1.DatabaseService.getDetectionBatches();
        return { ok: true, batches };
    }
    catch (error) {
        return { ok: false, error: 'getDetectionBatches failed: ' + error };
    }
}
async function updateDetectionBatchName(id, name) {
    try {
        database_1.DatabaseService.updateDetectionBatchName(id, name);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'updateDetectionBatchName failed: ' + error };
    }
}
async function deleteDetectionBatch(id) {
    try {
        database_1.DatabaseService.deleteDetectionBatch(id);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'deleteDetectionBatch failed: ' + error };
    }
}
// --- Detections ---
async function getDetectionsForBatch(batchId, species, minConfidence) {
    try {
        const detections = database_1.DatabaseService.getDetectionsForBatch(batchId, species, minConfidence);
        return { ok: true, detections };
    }
    catch (error) {
        return { ok: false, error: 'getDetectionsForBatch failed: ' + error };
    }
}
async function getAvailableSpecies() {
    try {
        const species = database_1.DatabaseService.getAvailableSpecies();
        return { ok: true, species };
    }
    catch (error) {
        return { ok: false, error: 'getAvailableSpecies failed: ' + error };
    }
}
async function updateDetectionLabel(id, label) {
    try {
        database_1.DatabaseService.updateDetectionLabel(id, label);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'updateDetectionLabel failed: ' + error };
    }
}
async function deleteDetection(id) {
    try {
        database_1.DatabaseService.deleteDetection(id);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'deleteDetection failed: ' + error };
    }
}
// --- Image Metadata ---
async function updateImageMetadata(id, metadata) {
    try {
        database_1.DatabaseService.updateImageMetadata(id, metadata);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'updateImageMetadata failed: ' + error };
    }
}
async function getImageMetadata(id) {
    try {
        const metadata = database_1.DatabaseService.getImageMetadata(id);
        return { ok: true, metadata };
    }
    catch (error) {
        return { ok: false, error: 'getImageMetadata failed: ' + error };
    }
}
// Function to read the JSON file and extract values for a specific key
const extractValuesForKey = async (filePath, key) => {
    try {
        // Read the JSON file
        const data = fs_extra_1.default.readFileSync(filePath, 'utf8');
        const jsonObject = JSON.parse(data);
        // Extract values for the specified key
        const values = jsonObject[key];
        // Check if values exist and return them, or return an empty array
        return Array.isArray(values) ? values : [];
    }
    catch (error) {
        console.error('Error reading or parsing JSON file:', error);
        return [];
    }
};
async function browseReidImage(date, time, group_id) {
    try {
        const userIdFolder = '1';
        const baseDir = path_1.default.join(userProfileDir, 'data/image_reid_output', userIdFolder);
        let targetDir, browseMode;
        if (!date) {
            browseMode = 'root';
            targetDir = path_1.default.resolve(baseDir); // Resolve the full path
            fs_extra_1.default.ensureDirSync(targetDir);
        }
        else if (!time) {
            browseMode = 'date';
            targetDir = path_1.default.resolve(baseDir, date); // Resolve the full path
        }
        else if (!group_id) {
            browseMode = 'time';
            const timeJson = time + '.json';
            let relDir = path_1.default.join(date, timeJson);
            targetDir = path_1.default.resolve(baseDir, relDir); // Resolve the full path
        }
        else {
            browseMode = 'group_id';
            const timeJson = time + '.json';
            let relDir = path_1.default.join(date, timeJson);
            targetDir = path_1.default.resolve(baseDir, relDir); // Resolve the full path
        }
        // Ensure the resolved path is still within the baseDir
        if (!targetDir.startsWith(baseDir)) {
            return { ok: false, error: 'Invalid folder path.' };
        }
        // Check if the directory exists before reading it
        if (!(await fs_extra_1.default.pathExists(targetDir))) {
            // console.log("browseMode: " + browseMode);
            return { ok: false, error: 'Directory not found.' };
        }
        if (browseMode === 'root') {
            const stat = fs_extra_1.default.statSync(targetDir);
            if (stat.isFile()) {
                return { ok: false, error: 'Path is a file, not a directory.' };
            }
            const files = await fs_extra_1.default.readdir(targetDir);
            const fileDetails = await Promise.all(files.map(async (file) => {
                const filePath = path_1.default.join(targetDir, file);
                const stat = await fs_extra_1.default.stat(filePath);
                return {
                    name: file,
                    isDirectory: stat.isDirectory(),
                    path: path_1.default.join(file),
                    date: file,
                    time: null,
                    group_id: null,
                    realDate: null,
                    realPath: null
                };
            }));
            return { ok: true, files: fileDetails };
        }
        else if (browseMode === 'date') {
            const stat = fs_extra_1.default.statSync(targetDir);
            if (stat.isFile()) {
                return { ok: false, error: 'Path is a file, not a directory.' };
            }
            const files = await fs_extra_1.default.readdir(targetDir);
            const fileDetails = await Promise.all(files.map(async (file) => {
                const filePath = path_1.default.join(targetDir, file);
                const basename = path_1.default.basename(filePath);
                const fileName = path_1.default.parse(basename).name; // Remove the extension
                return {
                    name: fileName,
                    isDirectory: true,
                    path: path_1.default.join(date, fileName),
                    date: date,
                    time: fileName,
                    group_id: null,
                    realDate: null,
                    realPath: null
                };
            }));
            return { ok: true, files: fileDetails };
        }
        else if (browseMode === 'time') {
            const ids = await extractKeysFromJson(targetDir);
            // console.log(ids);
            return {
                ok: true,
                files: ids.map((key) => ({
                    name: key,
                    isDirectory: true,
                    path: path_1.default.join(date, time, key),
                    date: date,
                    time: time,
                    group_id: key,
                    realDate: null,
                    realPath: null
                }))
            };
        }
        else if (browseMode === 'group_id') {
            // Extract values
            const imagePaths = await extractValuesForKey(targetDir, group_id);
            // Extract filenames from the paths
            // const imageNames = imagePaths.map(imagePath => path.basename(imagePath));
            return {
                ok: true,
                files: imagePaths.map((key) => ({
                    name: path_1.default.basename(key),
                    isDirectory: false,
                    path: path_1.default.join(date, time, group_id, path_1.default.basename(key)),
                    date: date,
                    time: time,
                    group_id: group_id,
                    realDate: key.split(path_1.default.sep)[0],
                    realPath: key.split(path_1.default.sep).slice(1).join(path_1.default.sep)
                }))
            };
        }
        else {
            return { ok: false, error: 'browseReidImage: Internal error related to browseMode.' };
        }
    }
    catch (error) {
        console.log(error);
        return { ok: false, error: 'browseReidImage: ' + error };
    }
}
async function deleteReidResult(date, time) {
    try {
        const userIdFolder = '1';
        const baseDir = path_1.default.join(userProfileDir, 'data/image_reid_output', userIdFolder);
        if (!date || !time) {
            return { ok: false, error: 'Missing one or more parameters: date, time.' };
        }
        const timeJson = time + '.json';
        const deteleDir = path_1.default.join(date, timeJson);
        const targetDir = path_1.default.resolve(baseDir, deteleDir);
        // Ensure the resolved path is still within the baseDir
        if (!targetDir.startsWith(baseDir)) {
            return { ok: false, error: 'Invalid path.' };
        }
        // Check if the target exists
        if (await fs_extra_1.default.pathExists(targetDir)) {
            await fs_extra_1.default.remove(targetDir); // Remove the file or directory
            // Check if the date folder is now empty
            const dateDir = path_1.default.join(targetDir, '..');
            const remainingFiles = await fs_extra_1.default.readdir(dateDir);
            if (remainingFiles.length === 0) {
                await fs_extra_1.default.remove(dateDir); // Remove the date folder if empty
            }
            return {
                ok: true,
                message: `ReID result (date = ${date}, time = ${time}) deleted successfully.`
            };
        }
        else {
            return { ok: false, error: `ReID result (date = ${date}, time = ${time}) not found.` };
        }
    }
    catch (error) {
        console.error(error);
        return { ok: false, error: 'deleteReidMessage: ' + error };
    }
}
async function renameReidGroup(date, time, old_group_id, new_group_id) {
    try {
        const userIdFolder = '1';
        const baseDir = path_1.default.join(userProfileDir, 'data/image_reid_output', userIdFolder);
        let targetDir;
        if (!date || !time || !old_group_id || !new_group_id) {
            return {
                ok: false,
                error: 'Missing one or more query parameters: date, time, old_group_id, new_group_id.'
            };
        }
        const timeJson = time + '.json';
        let relDir = path_1.default.join(date, timeJson);
        targetDir = path_1.default.resolve(baseDir, relDir); // Resolve the full path
        // Ensure the resolved path is still within the baseDir
        if (!targetDir.startsWith(baseDir)) {
            return { ok: false, error: 'Invalid folder path.' };
        }
        // Check if the directory exists before reading it
        if (!(await fs_extra_1.default.pathExists(targetDir))) {
            return { ok: false, error: 'Directory not found.' };
        }
        // Read the JSON file
        const fileData = await fs_extra_1.default.readJson(targetDir);
        // Check if old_group_id exists
        if (!fileData.hasOwnProperty(old_group_id)) {
            return { ok: false, error: `Key "${old_group_id}" not found.` };
        }
        // Check if new_group_id already exists
        if (fileData.hasOwnProperty(new_group_id)) {
            if (new_group_id === old_group_id) {
                return {
                    ok: true,
                    message: 'The new name is the same as the old name. The group name will not change. '
                };
            }
            return {
                ok: false,
                message: `Key "${new_group_id}" already exists. Chose a different name.`
            };
        }
        // Create a new object to maintain the original order of keys
        const newData = {};
        // Loop through the existing keys in fileData
        Object.keys(fileData).forEach((key) => {
            // If the key is the old_group_id, add it to newData with the new_group_id
            if (key === old_group_id) {
                newData[new_group_id] = fileData[old_group_id];
            }
            else {
                // Otherwise, just copy the existing key-value pair
                newData[key] = fileData[key];
            }
        });
        // Write the modified JSON back to the file
        await fs_extra_1.default.writeJson(targetDir, newData, { spaces: 4 });
        return { ok: true, message: `Successfully renamed from ${old_group_id} to ${new_group_id}.` };
    }
    catch (error) {
        console.error(error);
        return { ok: false, error: 'renameReidGroup: ' + error };
    }
}
function terminateAI() {
    (0, python_1.terminateSubprocess)();
}
async function checkIsDirectory(filePath) {
    try {
        const stat = await fs_extra_1.default.stat(filePath);
        return stat.isDirectory();
    }
    catch (error) {
        console.error('Error checking directory:', error);
        return false;
    }
}
async function openFileDialog() {
    const result = await electron_1.dialog.showOpenDialog({
        properties: ['openFile', 'openDirectory', 'multiSelections'],
        filters: [
            { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] }
        ]
    });
    if (result.canceled) {
        return { canceled: true, filePaths: [] };
    }
    else {
        return { canceled: false, filePaths: result.filePaths };
    }
}
async function saveImages(sourcePaths) {
    try {
        const result = await electron_1.dialog.showOpenDialog({
            title: 'Select Destination Folder',
            properties: ['openDirectory', 'createDirectory']
        });
        if (result.canceled || result.filePaths.length === 0) {
            return { ok: false, error: 'Operation canceled' };
        }
        const destDir = result.filePaths[0];
        let successCount = 0;
        let failCount = 0;
        for (const srcPath of sourcePaths) {
            try {
                const fileName = path_1.default.basename(srcPath);
                const destPath = path_1.default.join(destDir, fileName);
                await fs_extra_1.default.copy(srcPath, destPath);
                successCount++;
            }
            catch (err) {
                console.error(`Failed to copy ${srcPath}:`, err);
                failCount++;
            }
        }
        return { ok: true, successCount, failCount };
    }
    catch (error) {
        console.error('Error saving images:', error);
        return { ok: false, error: String(error) };
    }
}
// ============================================================================
// Smart ReID (New DB-based system)
// ============================================================================
/**
 * Smart ReID - Add re-identification job to queue
 * Uses the job manager for proper async handling
 * If images need detection first, queues a detect job that chains to reid
 */
async function smartReID(imageIds, species) {
    try {
        if (!imageIds || !Array.isArray(imageIds) || imageIds.length === 0) {
            return { ok: false, error: 'No images selected.' };
        }
        if (!species) {
            return { ok: false, error: 'No species selected.' };
        }
        // Check if any images need detection first
        const imagesWithoutDetections = database_1.DatabaseService.getImagesWithoutDetections(imageIds);
        console.log(`[smartReID] imageIds: ${JSON.stringify(imageIds)}`);
        console.log(`[smartReID] imagesWithoutDetections: ${JSON.stringify(imagesWithoutDetections)}`);
        if (imagesWithoutDetections.length > 0) {
            // Get paths for images that need detection
            const images = database_1.DatabaseService.getImagesByIds(imagesWithoutDetections);
            const selectedPaths = images.map(img => img.original_path);
            console.log(`[smartReID] Queuing detect job for ${selectedPaths.length} images, then reid`);
            // Queue a detect job that will chain to reid
            jobs_1.JobManager.getInstance().addJob('detect', {
                selectedPaths,
                chainToReid: true,
                imageIds,
                species
            });
        }
        else {
            console.log(`[smartReID] All images have detections, queuing reid directly`);
            // All images have detections, queue reid directly
            jobs_1.JobManager.getInstance().addJob('reid', { imageIds, species });
        }
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'smartReID failed: ' + error };
    }
}
// --- ReID Run Management ---
async function getReidRuns() {
    try {
        const runs = database_1.DatabaseService.getReidRunsWithStats();
        return { ok: true, runs };
    }
    catch (error) {
        return { ok: false, error: 'getReidRuns failed: ' + error };
    }
}
async function getReidRun(id) {
    try {
        const run = database_1.DatabaseService.getReidRun(id);
        if (!run) {
            return { ok: false, error: 'ReID run not found' };
        }
        return { ok: true, run };
    }
    catch (error) {
        return { ok: false, error: 'getReidRun failed: ' + error };
    }
}
async function deleteReidRunById(id) {
    try {
        database_1.DatabaseService.deleteReidRun(id);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'deleteReidRun failed: ' + error };
    }
}
async function updateReidRunName(id, name) {
    try {
        database_1.DatabaseService.updateReidRunName(id, name);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'updateReidRunName failed: ' + error };
    }
}
// --- ReID Results (Paginated) ---
async function getReidResults(filter) {
    try {
        const result = database_1.DatabaseService.getReidResults(filter);
        if (!result) {
            return { ok: false, error: 'ReID run not found' };
        }
        return { ok: true, result };
    }
    catch (error) {
        return { ok: false, error: 'getReidResults failed: ' + error };
    }
}
// Get ReID results for a single image (all runs/individuals that include this image)
async function getReidResultsForImage(imageId) {
    try {
        const results = database_1.DatabaseService.getReidResultsForImage(imageId);
        return { ok: true, results };
    }
    catch (error) {
        return { ok: false, error: 'getReidResultsForImage failed: ' + error };
    }
}
// Get ReID results for multiple images
async function getReidResultsForImages(imageIds) {
    try {
        const results = database_1.DatabaseService.getReidResultsForImages(imageIds);
        return { ok: true, results };
    }
    catch (error) {
        return { ok: false, error: 'getReidResultsForImages failed: ' + error };
    }
}
// Get latest detections for images (only from most recent batch per image)
async function getLatestDetectionsForImages(imageIds) {
    try {
        const detections = database_1.DatabaseService.getLatestDetectionsForImages(imageIds);
        return { ok: true, detections };
    }
    catch (error) {
        return { ok: false, error: 'getLatestDetectionsForImages failed: ' + error };
    }
}
// --- ReID Individual Management ---
async function updateReidIndividualName(id, displayName) {
    try {
        database_1.DatabaseService.updateReidIndividualName(id, displayName);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'updateReidIndividualName failed: ' + error };
    }
}
async function updateReidIndividualColor(id, color) {
    try {
        database_1.DatabaseService.updateReidIndividualColor(id, color);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'updateReidIndividualColor failed: ' + error };
    }
}
async function mergeReidIndividuals(targetId, sourceIds) {
    try {
        database_1.DatabaseService.mergeReidIndividuals(targetId, sourceIds);
        return { ok: true };
    }
    catch (error) {
        return { ok: false, error: 'mergeReidIndividuals failed: ' + error };
    }
}
async function getDashboardStats() {
    try {
        const stats = database_1.DatabaseService.getDashboardStats();
        return { ok: true, stats };
    }
    catch (error) {
        return { ok: false, error: 'getDashboardStats failed: ' + error };
    }
}
// --- Embeddings Cache ---
async function clearEmbeddingsCache() {
    try {
        const count = database_1.DatabaseService.clearAllEmbeddings();
        return { ok: true, count };
    }
    catch (error) {
        return { ok: false, error: 'clearEmbeddingsCache failed: ' + error };
    }
}
