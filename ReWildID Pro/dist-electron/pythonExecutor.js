"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.executePythonCode = executePythonCode;
const child_process_1 = require("child_process");
const electron_1 = require("electron");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
/**
 * Get the path to the Python executable in the CARE venv.
 */
function getPythonPath() {
    // In dev mode, use the local venv
    if (!electron_1.app.isPackaged) {
        const venvPath = path.resolve(__dirname, '../../python/.venv');
        if (fs.existsSync(venvPath)) {
            if (os.platform() === 'win32') {
                return path.join(venvPath, 'Scripts', 'python.exe');
            }
            else {
                return path.join(venvPath, 'bin', 'python');
            }
        }
    }
    else {
        // In packaged mode, use the bundled Python
        const ext = os.platform() === 'win32' ? '.exe' : '';
        const bundledPython = path.join(process.resourcesPath, 'app.asar.unpacked', 'resources', 'care-detect-reid', `care-detect-reid${ext}`);
        if (fs.existsSync(bundledPython)) {
            return bundledPython;
        }
    }
    // Fallback to system Python
    return os.platform() === 'win32' ? 'python' : 'python3';
}
/**
 * Execute Python code locally using the CARE Python environment.
 * Captures stdout, stderr, and any matplotlib figures saved.
 */
async function executePythonCode(code) {
    return new Promise((resolve) => {
        // Create a temp directory for this execution
        const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rewildid-py-'));
        const scriptPath = path.join(tempDir, 'script.py');
        // Get the database path
        const dbPath = path.join(process.cwd(), 'data', 'library.db').replace(/\\/g, '/');
        // Wrap the code to save matplotlib figures automatically
        const wrappedCode = `
import sys
import os

# Database path for querying RewildID data
DB_PATH = ${JSON.stringify(dbPath)}

# Set the temp directory for saving figures
_TEMP_DIR = ${JSON.stringify(tempDir)}

# Patch matplotlib to save figures automatically
_saved_figures = []

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    _original_show = plt.show
    def _patched_show(*args, **kwargs):
        fig = plt.gcf()
        fig_path = os.path.join(_TEMP_DIR, f'figure_{len(_saved_figures)}.png')
        fig.savefig(fig_path, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        _saved_figures.append(fig_path)
        plt.close(fig)
    
    plt.show = _patched_show
except ImportError:
    pass

# Execute the user's code
try:
${code.split('\n').map(line => '    ' + line).join('\n')}
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)

# Print saved figure paths for parsing
if _saved_figures:
    print("__FIGURES__:" + ",".join(_saved_figures))
`;
        fs.writeFileSync(scriptPath, wrappedCode);
        const pythonPath = getPythonPath();
        console.log(`[Python] Using Python at: ${pythonPath}`);
        let stdout = '';
        let stderr = '';
        const pythonProcess = (0, child_process_1.spawn)(pythonPath, [scriptPath], {
            cwd: tempDir,
            env: { ...process.env },
        });
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        pythonProcess.on('close', (exitCode) => {
            const images = [];
            let cleanOutput = stdout;
            // Extract figure paths from output
            const figureMatch = stdout.match(/__FIGURES__:(.+)/);
            if (figureMatch) {
                const figurePaths = figureMatch[1].split(',');
                for (const figPath of figurePaths) {
                    try {
                        const imageData = fs.readFileSync(figPath.trim());
                        images.push(`data:image/png;base64,${imageData.toString('base64')}`);
                    }
                    catch (e) {
                        console.error('Failed to read figure:', e);
                    }
                }
                // Remove the __FIGURES__ line from output
                cleanOutput = stdout.replace(/__FIGURES__:.+\n?/, '').trim();
            }
            // Cleanup temp directory
            try {
                fs.rmSync(tempDir, { recursive: true, force: true });
            }
            catch (e) {
                console.error('Failed to cleanup temp dir:', e);
            }
            console.log(`[Python] Execution complete. Exit code: ${exitCode}, Images: ${images.length}`);
            resolve({
                success: exitCode === 0,
                output: cleanOutput,
                error: stderr || null,
                images,
            });
        });
        pythonProcess.on('error', (err) => {
            // Cleanup temp directory
            try {
                fs.rmSync(tempDir, { recursive: true, force: true });
            }
            catch (e) {
                console.error('Failed to cleanup temp dir:', e);
            }
            resolve({
                success: false,
                output: '',
                error: `Failed to start Python: ${err.message}. The CARE Python environment may not be set up.`,
                images: [],
            });
        });
        // Set a timeout of 60 seconds
        setTimeout(() => {
            pythonProcess.kill();
            resolve({
                success: false,
                output: stdout,
                error: 'Execution timed out after 60 seconds',
                images: [],
            });
        }, 60000);
    });
}
