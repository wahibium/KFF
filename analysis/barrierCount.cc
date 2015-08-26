{   
    unsigned long warpId = (blockId() * blockDim() + blockThreadId()) >> 5;

    ON_INSTRUCTION:
    BARRIER:
    {
        
        if(leastActiveThreadInWarp())
        {
            globalMem[warpId * 2] = globalMem[warpId * 2] + 1; 
        }
    }
}
