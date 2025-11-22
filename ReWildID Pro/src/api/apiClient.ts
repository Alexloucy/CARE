// Mock API client
const apiClient = {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    get: async (_url: string) => ({ data: {} }),
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    post: async (_url: string, _data: any) => ({ data: {} }),
};

export default apiClient;
