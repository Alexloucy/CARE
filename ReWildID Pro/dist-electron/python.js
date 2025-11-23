"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.terminateSubprocess = terminateSubprocess;
exports.setSubProcess = setSubProcess;
exports.spawnPythonSubprocess = spawnPythonSubprocess;
const node_child_process_1 = require("node:child_process");
const electron_1 = require("electron");
const path_1 = __importDefault(require("path"));
const os_1 = __importDefault(require("os"));
const fs_extra_1 = __importDefault(require("fs-extra"));
let subProcess = null;
function terminateSubprocess() {
    // Terminate any running AI process.
    if (subProcess === null) {
        return;
    }
    subProcess.kill();
    subProcess = null;
}
function setSubProcess(ps) {
    subProcess = ps;
}
function conda() {
    try {
        const ps = (0, node_child_process_1.spawnSync)('conda info');
        return ps.status !== undefined && ps.status == 0;
    }
    catch (e) {
        return false;
    }
}
function spawnPythonSubprocess(args) {
    let ps = null;
    let python = '';
    console.log(`process.resourcesPath=${process.resourcesPath}`);
    if (process.env.PYTHON_SCRIPT_PATH) {
        if (process.env.VIRTUAL_ENV) {
            // Standard Python virtual env.
            if (os_1.default.platform() == 'win32') {
                python = path_1.default.join(process.env.VIRTUAL_ENV, 'Scripts', 'python');
            }
            else {
                python = path_1.default.join(process.env.VIRTUAL_ENV, 'bin', 'python');
            }
            args = [process.env.PYTHON_SCRIPT_PATH, ...args];
            console.log(`Spawning Python subprocess using venv.`);
        }
        else if (conda()) {
            const scriptPath = process.env.PYTHON_SCRIPT_PATH;
            const condaEnv = process.env.DEVICE == 'GPU' ? 'CARE-GPU' : 'CARE';
            python = os_1.default.platform() == 'win32' ? 'python' : 'python3';
            args = ['run', '--no-capture-output', '-n', condaEnv, python, scriptPath].concat(args);
            console.log(`Spawning Conda Python subprocess.`);
        }
    }
    else {
        if (electron_1.app.isPackaged) {
            console.log('Running Pyinstaller Python');
            const ext = os_1.default.platform() == 'win32' ? '.exe' : '';
            python = path_1.default.join(process.resourcesPath, 'app.asar.unpacked', 'resources', 'care-detect-reid', `care-detect-reid${ext}`);
        }
        else {
            console.log('Running Dev Mode Python');
            // Adjusted path for ReWildID Pro structure
            // Assuming python.ts is in electron/, same as controller.ts was
            const pythonScriptPath = path_1.default.resolve(__dirname, '../../python/main.py');
            const venvPath = path_1.default.resolve(__dirname, '../../python/.venv');
            args = [pythonScriptPath, ...args];
            if (fs_extra_1.default.existsSync(venvPath)) {
                if (os_1.default.platform() == 'win32') {
                    python = path_1.default.join(venvPath, 'Scripts', 'python.exe');
                }
                else {
                    python = path_1.default.join(venvPath, 'bin', 'python');
                }
                console.log(`Using local venv at: ${python}`);
            }
            else {
                python = 'python'; // Fallback to global python
                console.log(`Using global python`);
            }
        }
    }
    console.log(`Spawn: ${python} ${args.join(' ')}`);
    try {
        ps = (0, node_child_process_1.spawn)(python, args);
    }
    catch (e) {
        console.log(e);
        throw e;
    }
    return ps;
}
