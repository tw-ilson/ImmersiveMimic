using UnityEngine;

/// <summary>
/// Example controller that uses SO101_IK to move the robot arm to follow a target sphere
/// Attach this to your robot arm base
/// </summary>
public class RobotArmController : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The target sphere object to follow")]
    public Transform targetSphere;

    [Header("Joint References")]
    [Tooltip("Assign your robot's articulation bodies in order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]")]
    public ArticulationBody[] joints = new ArticulationBody[5];

    private SO101_IK ikSolver;

    void Start()
    {
        // Get or add the IK solver component
        ikSolver = GetComponent<SO101_IK>();
        if (ikSolver == null)
        {
            ikSolver = gameObject.AddComponent<SO101_IK>();
        }
    }

    void Update()
    {
        if (targetSphere == null)
        {
            Debug.LogWarning("Target sphere not assigned!");
            return;
        }

        // Get target position relative to robot base
        Vector3 localTarget = transform.InverseTransformPoint(targetSphere.position);

        // Calculate IK
        float[] jointAngles;
        bool success = ikSolver.CalculateIK(localTarget.x, localTarget.y, localTarget.z, out jointAngles);

        if (success)
        {
            // Apply joint angles to the robot
            ApplyJointAngles(jointAngles);
        }
    }

    /// <summary>
    /// Apply calculated joint angles to the robot's articulation bodies
    /// </summary>
    void ApplyJointAngles(float[] angles)
    {
        for (int i = 0; i < joints.Length && i < angles.Length; i++)
        {
            if (joints[i] != null)
            {
                // Convert angle to ArticulationDrive target
                var drive = joints[i].xDrive;
                drive.target = angles[i];
                joints[i].xDrive = drive;
            }
        }
    }

    /// <summary>
    /// Visualize the target and current end effector position
    /// </summary>
    void OnDrawGizmos()
    {
        if (targetSphere != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawWireSphere(targetSphere.position, 0.02f);

            // Draw line from base to target
            Gizmos.color = Color.cyan;
            Gizmos.DrawLine(transform.position, targetSphere.position);
        }
    }
}
