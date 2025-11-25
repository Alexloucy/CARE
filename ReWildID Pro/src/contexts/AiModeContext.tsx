import React, { createContext, useContext, useState } from 'react';

interface AiModeContextType {
    shouldPlayEffect: boolean;
    setShouldPlayEffect: (value: boolean) => void;
}

export const AiModeContext = createContext<AiModeContextType | undefined>(undefined);

export const AiModeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [shouldPlayEffect, setShouldPlayEffect] = useState(false);

    return (
        <AiModeContext.Provider value={{ shouldPlayEffect, setShouldPlayEffect }}>
            {children}
        </AiModeContext.Provider>
    );
};

export const useAiMode = () => {
    const context = useContext(AiModeContext);
    if (context === undefined) {
        return { shouldPlayEffect: false, setShouldPlayEffect: () => { } }; // Fallback for when used outside provider
    }
    return context;
};
