import { spawn, ChildProcess, spawnSync } from 'node:child_process';
import { app } from 'electron';
import path from 'path';
import os from 'os';
import fs from 'fs-extra';

let subProcess: ChildProcess | null = null;

export function terminateSubprocess() {
    // Terminate any running AI process.
    if (subProcess === null) {
        return;
    }
    subProcess.kill();
    subProcess = null;
}

export function setSubProcess(ps: ChildProcess | null) {
    subProcess = ps;
}

function conda(): boolean {
    try {
        const ps = spawnSync('conda info');
        return ps.status !== undefined && ps.status == 0;
    } catch (e) {
        return false;
    }
}

export function spawnPythonSubprocess(args: string[]) {
    let ps: ChildProcess | null = null;
    let python = '';

    console.log(`process.resourcesPath=${process.resourcesPath}`);
    if (process.env.PYTHON_SCRIPT_PATH) {
        if (process.env.VIRTUAL_ENV) {
            // Standard Python virtual env.
            if (os.platform() == 'win32') {
                python = path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python');
            } else {
                python = path.join(process.env.VIRTUAL_ENV, 'bin', 'python');
            }
            args = [process.env.PYTHON_SCRIPT_PATH, ...args];
            console.log(`Spawning Python subprocess using venv.`);
        } else if (conda()) {
            const scriptPath = process.env.PYTHON_SCRIPT_PATH;
            const condaEnv = process.env.DEVICE == 'GPU' ? 'CARE-GPU' : 'CARE';
            python = os.platform() == 'win32' ? 'python' : 'python3';
            args = ['run', '--no-capture-output', '-n', condaEnv, python, scriptPath].concat(args);
            console.log(`Spawning Conda Python subprocess.`);
        }
    } else {
        if (app.isPackaged) {
            console.log('Running Pyinstaller Python');
            const ext = os.platform() == 'win32' ? '.exe' : '';
            python = path.join(
                process.resourcesPath,
                'app.asar.unpacked',
                'resources',
                'care-detect-reid',
                `care-detect-reid${ext}`
            );
        } else {
            console.log('Running Dev Mode Python');

            // Adjusted path for ReWildID Pro structure
            // Assuming python.ts is in electron/, same as controller.ts was
            const pythonScriptPath = path.resolve(__dirname, '../../python/main.py');
            const venvPath = path.resolve(__dirname, '../../python/.venv');

            args = [pythonScriptPath, ...args];

            if (fs.existsSync(venvPath)) {
                if (os.platform() == 'win32') {
                    python = path.join(venvPath, 'Scripts', 'python.exe');
                } else {
                    python = path.join(venvPath, 'bin', 'python');
                }
                console.log(`Using local venv at: ${python}`);
            } else {
                python = 'python'; // Fallback to global python
                console.log(`Using global python`);
            }
        }
    }
    console.log(`Spawn: ${python} ${args.join(' ')}`);
    try {
        ps = spawn(python, args);
    } catch (e) {
        console.log(e);
        throw e;
    }
    return ps;
}
