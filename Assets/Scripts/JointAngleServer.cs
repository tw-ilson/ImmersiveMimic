using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// Sends joint angle data via UDP to a specified IP address and port
/// Attach this to the same GameObject as SO101_IK or reference it
/// </summary>
public class JointAngleServer : MonoBehaviour
{
    [Header("Network Configuration")]
    [Tooltip("Target IP address to send UDP packets to")]
    public string targetIP = "192.168.1.100";

    [Tooltip("Target port to send UDP packets to")]
    public int targetPort = 8888;

    [Header("Send Settings")]
    [Tooltip("How often to send updates (seconds). 0 = every frame")]
    public float sendInterval = 0.05f; // 20Hz default

    [Header("Data Format")]
    [Tooltip("Send as JSON (true) or comma-separated values (false)")]
    public bool sendAsJSON = true;

    [Header("References")]
    [Tooltip("Reference to the SO101_IK component")]
    public SO101_IK ikSolver;

    private UdpClient udpClient;
    private IPEndPoint targetEndPoint;
    private float timeSinceLastSend = 0f;
    private bool isInitialized = false;

    void Start()
    {
        // Get SO101_IK component if not assigned
        if (ikSolver == null)
        {
            ikSolver = GetComponent<SO101_IK>();
            if (ikSolver == null)
            {
                Debug.LogError("JointAngleServer: No SO101_IK component found!");
                enabled = false;
                return;
            }
        }

        InitializeUDP();
    }

    void InitializeUDP()
    {
        try
        {
            udpClient = new UdpClient();
            targetEndPoint = new IPEndPoint(IPAddress.Parse(targetIP), targetPort);
            isInitialized = true;
            Debug.Log($"UDP sender initialized: {targetIP}:{targetPort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize UDP client: {e.Message}");
            isInitialized = false;
        }
    }

    void Update()
    {
        if (!isInitialized || ikSolver == null || ikSolver.jointAngles == null)
            return;

        timeSinceLastSend += Time.deltaTime;

        // Send at specified interval (or every frame if interval is 0)
        if (sendInterval <= 0f || timeSinceLastSend >= sendInterval)
        {
            SendJointAngles();
            timeSinceLastSend = 0f;
        }
    }

    void SendJointAngles()
    {
        try
        {
            string message;

            if (sendAsJSON)
            {
                // Create JSON message
                JointAngleMessage jsonMsg = new JointAngleMessage
                {
                    timestamp = Time.time,
                    jointAngles = ikSolver.jointAngles
                };
                message = JsonUtility.ToJson(jsonMsg);
            }
            else
            {
                // Create CSV message: timestamp,j0,j1,j2,j3,j4
                message = $"{Time.time:F3},{string.Join(",", Array.ConvertAll(ikSolver.jointAngles, x => x.ToString("F2")))}";
            }

            byte[] data = Encoding.UTF8.GetBytes(message);
            udpClient.Send(data, data.Length, targetEndPoint);

            // Optional: uncomment for debugging
            // Debug.Log($"Sent UDP: {message}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to send UDP packet: {e.Message}");
        }
    }

    void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
            Debug.Log("UDP sender closed");
        }
    }

    void OnApplicationQuit()
    {
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }

    /// <summary>
    /// Update the target IP address at runtime
    /// </summary>
    public void SetTargetIP(string ip)
    {
        targetIP = ip;
        if (isInitialized)
        {
            targetEndPoint = new IPEndPoint(IPAddress.Parse(targetIP), targetPort);
            Debug.Log($"Target IP updated to: {targetIP}");
        }
    }

    /// <summary>
    /// Update the target port at runtime
    /// </summary>
    public void SetTargetPort(int port)
    {
        targetPort = port;
        if (isInitialized)
        {
            targetEndPoint = new IPEndPoint(IPAddress.Parse(targetIP), targetPort);
            Debug.Log($"Target port updated to: {targetPort}");
        }
    }
}

/// <summary>
/// Data structure for JSON serialization of joint angles
/// </summary>
[Serializable]
public class JointAngleMessage
{
    public float timestamp;
    public float[] jointAngles;
}
