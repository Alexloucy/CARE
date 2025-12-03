export interface ReidRun {
    id: number;
    name: string;
    species: string;
    created_at: number;
    individual_count: number;
    detection_count: number;
}

export interface ReidDetection {
    id: number;
    image_id: number;
    label: string;
    confidence: number;
    detection_confidence: number;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    source: string;
    batch_id: number;
    created_at: number;
    image_path: string;
    image_preview_path?: string;
}

export interface ReidIndividual {
    id: number;
    run_id: number;
    name: string;
    display_name: string;
    color: string;
    created_at: number;
    member_count: number;
    detections: ReidDetection[];
}
