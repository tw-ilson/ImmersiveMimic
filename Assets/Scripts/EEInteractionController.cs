using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Controller for EEInteraction GameObject behaviors
/// Handles grab/release events and coordinates multiple behaviors
/// Attach this to the EEInteraction GameObject
/// </summary>
public class EEInteractionController : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Reference to the gripper_link transform on the robot")]
    public Transform gripperLink;

    [Tooltip("Reference to the VisualSphere's MeshRenderer")]
    public MeshRenderer visualRenderer;

    [Header("Visual Feedback")]
    [Tooltip("Color when not being grabbed")]
    public Color normalColor = Color.blue;

    [Tooltip("Color when being grabbed")]
    public Color grabbedColor = Color.red;

    private Grabbable grabbable;
    private Material visualMaterial;
    private bool isGrabbed = false;

    void Start()
    {
        // Position at gripper_link
        if (gripperLink != null)
        {
            transform.position = gripperLink.position;
            Debug.Log($"EEInteraction positioned at gripper_link: {gripperLink.position}");
        }
        else
        {
            Debug.LogWarning("EEInteractionController: gripperLink reference not assigned!");
        }

        // Get the Grabbable component
        grabbable = GetComponent<Grabbable>();
        if (grabbable == null)
        {
            Debug.LogError("EEInteractionController: No Grabbable component found on this GameObject!");
            enabled = false;
            return;
        }

        // Find VisualSphere if not assigned
        if (visualRenderer == null)
        {
            Transform visualSphere = transform.Find("VisualSphere");
            if (visualSphere != null)
            {
                visualRenderer = visualSphere.GetComponent<MeshRenderer>();
            }

            if (visualRenderer == null)
            {
                Debug.LogError("EEInteractionController: Could not find VisualSphere MeshRenderer!");
                enabled = false;
                return;
            }
        }

        // Get or create material instance
        visualMaterial = visualRenderer.material;

        // Initialize visual state
        InitializeVisuals();

        // Subscribe to grab events
        grabbable.WhenPointerEventRaised += HandlePointerEvent;
    }

    void OnDestroy()
    {
        // Unsubscribe from events
        if (grabbable != null)
        {
            grabbable.WhenPointerEventRaised -= HandlePointerEvent;
        }
    }

    private void HandlePointerEvent(PointerEvent evt)
    {
        switch (evt.Type)
        {
            case PointerEventType.Select:
                OnGrab();
                break;

            case PointerEventType.Unselect:
                OnRelease();
                break;
        }
    }

    /// <summary>
    /// Initialize visual state on start
    /// </summary>
    private void InitializeVisuals()
    {
        visualMaterial.color = normalColor;
    }

    /// <summary>
    /// Called when the object is grabbed
    /// Add additional grab behaviors here
    /// </summary>
    private void OnGrab()
    {
        isGrabbed = true;

        // Visual feedback: change color to red
        visualMaterial.color = grabbedColor;

        Debug.Log("EEInteraction grabbed");

        // TODO: Add additional grab behaviors here
        // Examples:
        // - Play haptic feedback
        // - Trigger sound effect
        // - Disable IK solver
        // - Record grab timestamp
        // - Send network event
    }

    /// <summary>
    /// Called when the object is released
    /// Add additional release behaviors here
    /// </summary>
    private void OnRelease()
    {
        isGrabbed = false;

        // Visual feedback: change color back to blue
        visualMaterial.color = normalColor;

        Debug.Log("EEInteraction released");

        // TODO: Add additional release behaviors here
        // Examples:
        // - Stop haptic feedback
        // - Trigger sound effect
        // - Re-enable IK solver
        // - Calculate grab duration
        // - Send network event
    }

    /// <summary>
    /// Get current grab state
    /// </summary>
    public bool IsGrabbed()
    {
        return isGrabbed;
    }
}
