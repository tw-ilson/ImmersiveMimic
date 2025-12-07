using UnityEngine;

/// <summary>
/// Inverse Kinematics solver for SO-101 robot arm
/// Based on the Python keyboard_control.py implementation
/// Takes x, y, z coordinates and calculates joint angles
/// </summary>
public class SO101_IK : MonoBehaviour
{
    [Header("Robot Configuration")]
    [Tooltip("Length of upper arm link (shoulder to elbow)")]
    public float l1 = 0.1159f; // Upper arm length (m)

    [Tooltip("Length of lower arm link (elbow to wrist)")]
    public float l2 = 0.1350f; // Lower arm length (m)

    [Header("End Effector Orientation")]
    [Tooltip("Pitch adjustment for end effector (degrees)")]
    public float pitchAdjustment = 0f;

    // Joint angle offsets from URDF geometry (calculated from Python code)
    private float theta1Offset;
    private float theta2Offset;

    private void Awake()
    {
        // Calculate offsets from URDF geometry (matching Python implementation)
        theta1Offset = Mathf.Atan2(0.028f, 0.11257f);
        theta2Offset = Mathf.Atan2(0.0052f, 0.1349f) + theta1Offset;
    }

    /// <summary>
    /// Solve inverse kinematics for target position (x, y, z)
    /// Returns joint angles in degrees: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
    /// </summary>
    /// <param name="targetX">X coordinate in meters</param>
    /// <param name="targetY">Y coordinate (height) in meters</param>
    /// <param name="targetZ">Z coordinate in meters</param>
    /// <param name="jointAngles">Output array of 5 joint angles in degrees</param>
    /// <returns>True if solution found, false if target unreachable</returns>
    public bool CalculateIK(float targetX, float targetY, float targetZ, out float[] jointAngles)
    {
        jointAngles = new float[5];

        // Step 1: Calculate shoulder_pan (joint1) from X-Z plane
        // This rotates the base to point toward the target in the horizontal plane
        float joint1 = Mathf.Atan2(targetX, targetZ) * Mathf.Rad2Deg;
        jointAngles[0] = joint1;

        // Step 2: Calculate distance in XZ plane for 2D IK problem
        // Now we solve the arm movement in the vertical plane
        float horizontalDist = Mathf.Sqrt(targetX * targetX + targetZ * targetZ);

        // Use horizontal distance and Y (height) for 2D IK
        float x = horizontalDist;
        float y = targetY;

        // Step 3: Solve 2D IK for shoulder_lift (joint2) and elbow_flex (joint3)
        float joint2Deg, joint3Deg;
        bool success = SolveIK2D(x, y, out joint2Deg, out joint3Deg);

        if (!success)
        {
            Debug.LogWarning($"Target position ({targetX}, {targetY}, {targetZ}) is unreachable");
            return false;
        }

        jointAngles[1] = joint2Deg;
        jointAngles[2] = joint3Deg;

        // Step 4: Calculate wrist_flex (joint4) to maintain end effector orientation
        // This keeps the end effector level, matching the Python implementation
        float joint4 = -joint2Deg - joint3Deg + pitchAdjustment;
        jointAngles[3] = joint4;

        // Step 5: wrist_roll (joint5) defaults to 0
        jointAngles[4] = 0f;

        return true;
    }

    /// <summary>
    /// Solve 2D inverse kinematics for a two-link arm
    /// This is a direct port of the Python inverse_kinematics function
    /// </summary>
    private bool SolveIK2D(float x, float y, out float joint2Deg, out float joint3Deg)
    {
        joint2Deg = 0f;
        joint3Deg = 0f;

        // Calculate distance from origin to target
        float r = Mathf.Sqrt(x * x + y * y);
        float rMax = l1 + l2; // Maximum reachable distance
        float rMin = Mathf.Abs(l1 - l2); // Minimum reachable distance

        // If target is beyond maximum workspace, scale it to the boundary
        if (r > rMax)
        {
            float scaleFactor = rMax / r;
            x *= scaleFactor;
            y *= scaleFactor;
            r = rMax;
        }
        // If target is less than minimum workspace, scale it
        else if (r < rMin && r > 0)
        {
            float scaleFactor = rMin / r;
            x *= scaleFactor;
            y *= scaleFactor;
            r = rMin;
        }

        // Use law of cosines to calculate theta2 (elbow angle)
        float cosTheta2 = -(r * r - l1 * l1 - l2 * l2) / (2f * l1 * l2);

        // Clamp to valid range to avoid NaN from acos
        cosTheta2 = Mathf.Clamp(cosTheta2, -1f, 1f);

        // Calculate theta2 (elbow angle)
        float theta2 = Mathf.PI - Mathf.Acos(cosTheta2);

        // Calculate theta1 (shoulder angle)
        float beta = Mathf.Atan2(y, x);
        float gamma = Mathf.Atan2(l2 * Mathf.Sin(theta2), l1 + l2 * Mathf.Cos(theta2));
        float theta1 = beta + gamma;

        // Convert theta1 and theta2 to joint2 and joint3 angles (add offsets from URDF)
        float joint2 = theta1 + theta1Offset;
        float joint3 = theta2 + theta2Offset;

        // Ensure angles are within URDF limits (in radians)
        joint2 = Mathf.Clamp(joint2, -0.1f, 3.45f);
        joint3 = Mathf.Clamp(joint3, -0.2f, Mathf.PI);

        // Convert from radians to degrees
        joint2Deg = joint2 * Mathf.Rad2Deg;
        joint3Deg = joint3 * Mathf.Rad2Deg;

        // Apply same transformations as Python code (lines 106-107)
        joint2Deg = 90f - joint2Deg;
        joint3Deg = joint3Deg - 90f;

        return true;
    }

    /// <summary>
    /// Get the maximum reach of the arm
    /// </summary>
    public float GetMaxReach()
    {
        return l1 + l2;
    }

    /// <summary>
    /// Get the minimum reach of the arm
    /// </summary>
    public float GetMinReach()
    {
        return Mathf.Abs(l1 - l2);
    }

    /// <summary>
    /// Visualize the workspace in the Scene view
    /// </summary>
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying)
        {
            // Calculate offsets if not initialized
            theta1Offset = Mathf.Atan2(0.028f, 0.11257f);
            theta2Offset = Mathf.Atan2(0.0052f, 0.1349f) + theta1Offset;
        }

        // Draw max reach sphere (yellow)
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, GetMaxReach());

        // Draw min reach sphere (red)
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(transform.position, GetMinReach());
    }
}
