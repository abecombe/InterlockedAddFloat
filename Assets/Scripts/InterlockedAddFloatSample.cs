using System;
using System.Linq;
using Abecombe.GPUUtil;
using UnityEngine;

public class InterlockedAddFloatSample : MonoBehaviour, IDisposable
{
    private GPUBuffer<float> _dataBuffer = new();
    private GPUBuffer<uint> _dataChunkBuffer = new();
    private GPUBuffer<uint> _uintSumBuffer = new();
    private GPUBuffer<float> _floatSumBuffer = new();

    private GPUComputeShader _interlockedAddFloatCs;

    private float[] _data;
    private uint[] _dataChunk;

    [SerializeField] private int _numData = 1000;
    [SerializeField] private int _numChunks = 10;

    #region Initialize Functions
    private void InitComputeShaders()
    {
        _interlockedAddFloatCs = new GPUComputeShader("InterlockedAddFloatCS");
    }

    private void InitBuffers()
    {
        _dataBuffer.Init(_numData);
        _dataChunkBuffer.Init(_numData);
        _uintSumBuffer.Init(_numChunks);
        _floatSumBuffer.Init(_numChunks);

        _data = new float[_numData];
        _dataChunk = new uint[_numData];
        for (int i = 0; i < _numData; i++)
        {
            _data[i] = UnityEngine.Random.value;
            _dataChunk[i] = (uint)UnityEngine.Random.Range(0, _numChunks);
        }
        _dataBuffer.SetData(_data);
        _dataChunkBuffer.SetData(_dataChunk);
        _uintSumBuffer.SetData(Enumerable.Repeat(0u, _numChunks).ToArray());
        _floatSumBuffer.SetData(Enumerable.Repeat(0f, _numChunks).ToArray());
    }

    private void InterlockedAddFloat()
    {
        var cs = _interlockedAddFloatCs;

        var k = cs.FindKernel("InterlockedAddFloat");
        k.SetBuffer("_DataBuffer", _dataBuffer);
        k.SetBuffer("_DataChunkBuffer", _dataChunkBuffer);
        k.SetBuffer("_UIntSumBuffer", _uintSumBuffer);
        k.Dispatch(_numData);

        k = cs.FindKernel("ConvertUIntToFloat");
        k.SetBuffer("_UIntSumBuffer", _uintSumBuffer);
        k.SetBuffer("_FloatSumBuffer", _floatSumBuffer);
        k.Dispatch(_numChunks);
    }

    private void CompareCpuGpuResults()
    {
        float[] cpuSum = Enumerable.Repeat(0f, _numChunks).ToArray();
        for (int i = 0; i < _numData; i++)
        {
            cpuSum[_dataChunk[i]] += _data[i];
        }

        float[] gpuSum = new float[_numChunks];
        _floatSumBuffer.GetData(gpuSum);

        for (int i = 0; i < _numChunks; i++)
        {
            Debug.Log($"**********************");
            Debug.Log($"Chunk {i}:");
            Debug.Log($"CPU Sum: {cpuSum[i]}");
            Debug.Log($"GPU Sum: {gpuSum[i]}");
            Debug.Log($"(CPU-GPU)/CPU: {(cpuSum[i] - gpuSum[i]) / cpuSum[i]}");
        }
    }
    #endregion

    #region Release Buffers
    public void Dispose()
    {
        _dataBuffer.Dispose();
        _dataChunkBuffer.Dispose();
        _uintSumBuffer.Dispose();
        _floatSumBuffer.Dispose();
    }
    #endregion

    #region MonoBehaviour
    private void OnEnable()
    {
        InitComputeShaders();
    }

    private void Start()
    {
        InitBuffers();
        InterlockedAddFloat();
        CompareCpuGpuResults();
    }

    private void OnDisable()
    {
        Dispose();
    }
    #endregion
}