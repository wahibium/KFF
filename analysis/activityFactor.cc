{
    unsigned long threadId = blockThreadId();
    unsigned long warpId = (blockId() * blockDim() + threadId) >> 5;
   
    ON_BASIC_BLOCK_EXIT:
    {
        if(leastActiveThreadInWarp())
        {
            globalMem[(warpId * 2)] =
                globalMem[(warpId * 2)] +  
                activeThreadCount() * basicBlockInstructionCount(); 
            globalMem[(warpId * 2) + 1] =
                globalMem[(warpId * 2) + 1] +  
                32 * basicBlockInstructionCount();             
        }
    }   
}

