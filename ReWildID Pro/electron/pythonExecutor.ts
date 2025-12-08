import { spawn } from 'child_process';
import { app } from 'electron';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

export interface PythonExecutionResult {
    success: boolean;
    output: string;
    error: string | null;
    images: string[];
}

/**
 * Get the path to the Python executable in the CARE venv.
 */
function getPythonPath(): string {
    // In dev mode, use the local venv
    if (!app.isPackaged) {
        const venvPath = path.resolve(__dirname, '../../python/.venv');

        if (fs.existsSync(venvPath)) {
            if (os.platform() === 'win32') {
                return path.join(venvPath, 'Scripts', 'python.exe');
            } else {
                return path.join(venvPath, 'bin', 'python');
            }
        }
    } else {
        // In packaged mode, use the bundled Python
        const ext = os.platform() === 'win32' ? '.exe' : '';
        const bundledPython = path.join(
            process.resourcesPath,
            'app.asar.unpacked',
            'resources',
            'care-detect-reid',
            `care-detect-reid${ext}`
        );
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
export async function executePythonCode(code: string): Promise<PythonExecutionResult> {
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

        const pythonProcess = spawn(pythonPath, [scriptPath], {
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
            const images: string[] = [];
            let cleanOutput = stdout;

            // Extract figure paths from output
            const figureMatch = stdout.match(/__FIGURES__:(.+)/);
            if (figureMatch) {
                const figurePaths = figureMatch[1].split(',');
                for (const figPath of figurePaths) {
                    try {
                        const imageData = fs.readFileSync(figPath.trim());
                        images.push(`data:image/png;base64,${imageData.toString('base64')}`);
                    } catch (e) {
                        console.error('Failed to read figure:', e);
                    }
                }
                // Remove the __FIGURES__ line from output
                cleanOutput = stdout.replace(/__FIGURES__:.+\n?/, '').trim();
            }

            // Cleanup temp directory
            try {
                fs.rmSync(tempDir, { recursive: true, force: true });
            } catch (e) {
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
            } catch (e) {
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
